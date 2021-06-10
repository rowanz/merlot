import argparse
import json
import pprint
import collections
import os
import sys
import tensorflow as tf
import subprocess
from PIL import Image
import numpy as np
import io
import hashlib
import tqdm
import itertools
from google.cloud import storage
from tempfile import TemporaryDirectory
from multiprocessing import Pool

sys.path.append('../../../')

# tokenizer for inputs
from utils.encode.encoder import get_encoder

_tokenizer = get_encoder()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_story_json')
    parser.add_argument('image_directory')
    parser.add_argument('--num_folds', default=64, type=int)
    parser.add_argument('--num_threads', default=8, type=int)
    parser.add_argument('--pad_to_batch_size', default=32, type=int)
    parser.add_argument('--just_one_perm', default=0, type=int, help='should we save just one permutation')
    parser.add_argument('--save_dir', default='', type=str, help='Directory to save files to')
    args = parser.parse_args()
    args.mode = args.input_story_json.split('/')[-1].split('.')[0].split('_')[0]
    return args


class GCSTFRecordWriter(object):
    def __init__(self, fn):
        self.fn = fn
        if fn.startswith('gs://'):
            self.s3client = None
            self.gclient = storage.Client()
            self.storage_dir = TemporaryDirectory()
            self.writer = tf.python_io.TFRecordWriter(os.path.join(self.storage_dir.name, 'temp.tfrecord'))
            self.bucket_name, self.file_name = self.fn.split('gs://', 1)[1].split('/', 1)

        else:
            self.s3client = None
            self.gclient = None
            self.bucket_name = None
            self.file_name = None
            self.storage_dir = None
            self.writer = tf.python_io.TFRecordWriter(fn)

    def write(self, x):
        self.writer.write(x)

    def close(self):
        self.writer.close()
        if self.gclient is not None:
            bucket = self.gclient.get_bucket(self.bucket_name)
            blob = bucket.blob(self.file_name)
            blob.upload_from_filename(os.path.join(self.storage_dir.name, 'temp.tfrecord'))
            self.storage_dir.cleanup()

    def __enter__(self):
        # Called when entering "with" context.
        return self

    def __exit__(self, *_):
        # Called when exiting "with" context.
        # Upload shit
        print("CALLING CLOSE")
        self.close()


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


from io import BytesIO
def convert_PIL_to_jpgstring(image):
    """
    :param image: Numpy array of an image [H, W, 3]
    :return: it, as a jpg string
    """
    if image.mode == "RGBA":
        image = image.convert("RGB")
    with BytesIO() as output:
        image.save(output, format='JPEG', quality=95)
        return output.getvalue()


def process(fname, image_directory, writer, fold, args):
    with open(fname) as f:
        data = json.load(f)

    # we need to make a flattened list of stories

    storyid2anns = collections.defaultdict(list)
    for ann in data['annotations']:
        ann = ann[0]
        storyid2anns[int(ann['story_id'])].append(ann)

    for k in storyid2anns.keys():
        assert len(storyid2anns[k]) == 5
        assert set([ann['worker_arranged_photo_order'] for ann in storyid2anns[k]]) == {0,1,2,3,4}
        storyid2anns[k] = sorted(storyid2anns[k], key=lambda x: x['worker_arranged_photo_order'])

    # get paths
    image_paths = os.listdir(image_directory)
    image_paths = {k.split('.')[0]: image_directory + '/' + k for k in image_paths}
        
    print('{} stories'.format(len(storyid2anns)))
    sorted_story_ids = list(sorted(storyid2anns.keys()))    
    n_written = 0

    to_write = []
    for idx, story_id in tqdm.tqdm(list(enumerate(sorted_story_ids))):
        
        # check if we should be doing this in the current batch.
        if idx % args.num_folds != fold: continue

        anns = storyid2anns[story_id]
        assert [a['worker_arranged_photo_order'] for a in anns] == [0,1,2,3,4]
        
        texts = [_tokenizer.encode(a['original_text']) for a in anns]
        images = [Image.open(image_paths[a['photo_flickr_id']]) for a in anns]
        for imidx, im in enumerate(images):
            try:
                im.thumbnail((800, 800))
            except:
                print('***INVALID IMAGE***'*10)
                pprint.pprint(anns[imidx])
                quit()

        jpgstrimages = []
        for imidx, im in enumerate(images):
            try:
                jpgstrimages.append(convert_PIL_to_jpgstring(im))
            except:
                print('***INVALID IMAGE***'*10)
                pprint.pprint(anns[imidx])
                quit()

        images = jpgstrimages
        image_hashes = [hashlib.sha256(im).hexdigest().encode('utf-8') for im in images]

        for perm in list(itertools.permutations(range(5))) if not args.just_one_perm else [[0,1,2,3,4]]:
            #9ABCDE
            perm_int_encode = int(''.join([str(x) for x in perm])) + 900000
            
            feature_dict = {
                'permutation_identity_encode': int64_feature(perm_int_encode),
                'story_id': int64_feature(story_id)}
                
            for perm_idx, perm_sorted_idx in enumerate(perm):
                feature_dict.update({
                    'sentence/sentence_{}'.format(perm_idx): int64_list_feature(texts[perm_sorted_idx]),
                    'image/image_{}_sha256'.format(perm_idx): bytes_feature(image_hashes[perm_sorted_idx]),
                    'image/image_{}_encoded'.format(perm_idx): bytes_feature(images[perm_sorted_idx]),
                    'image/image_{}_format'.format(perm_idx): bytes_feature('jpeg'.encode('utf8')),
                    'image/image_{}_is_valid'.format(perm_idx): int64_feature(1)})

            example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
            writer.write(example.SerializeToString())
            n_written += 1

    print('Wrote {}'.format(n_written))
    return n_written, example


def _process(idx_args):
    idx, args = idx_args
    if args.just_one_perm:
        output_name = '{}-justone-{:05d}-of-{:05d}.tfrecord'.format(args.save_dir + '{}'.format(args.mode), idx, args.num_folds)
    else:
        output_name = '{}-{:05d}-of-{:05d}.tfrecord'.format(args.save_dir + '{}'.format(args.mode), idx, args.num_folds)

    print('writing to {}'.format(output_name))
    writer = GCSTFRecordWriter(output_name)
    count, _ = process(args.input_story_json,
                       args.image_directory,
                       writer,
                       idx,
                       args)
    writer.close()
    return count


def _process_final(idx_args_total):
    idx, args, total = idx_args_total
    if args.just_one_perm:
        output_name = '{}-justone-{:05d}-of-{:05d}.tfrecord'.format(args.save_dir + '{}'.format(args.mode), idx, args.num_folds)
    else:
        output_name = '{}-{:05d}-of-{:05d}.tfrecord'.format(args.save_dir + '{}'.format(args.mode), idx, args.num_folds)
    print('writing to {}'.format(output_name))
    writer = GCSTFRecordWriter(output_name)
    count, final_ex = process(args.input_story_json,
                              args.image_directory,
                              writer,
                              idx,
                              args)
    n_to_add = args.pad_to_batch_size - (total + count) % args.pad_to_batch_size
    if args.mode != 'train' and n_to_add > 0:
        print('we are not in training, so we will add {} to make batch size {} work.'.format(
            n_to_add, args.pad_to_batch_size))
        
        for _ in range(n_to_add):
            writer.write(final_ex.SerializeToString())
            count += 1

    writer.close()

    if args.mode == 'train':
        return count, 0
    else:
        return count, n_to_add


def main():
    args = parse_args()

    with Pool(args.num_threads) as p:
        res = p.map(_process, [(idx, args) for idx in range(args.num_folds-1)])
    print('{} total examples written in parallel'.format(np.sum(res)))

    total = int(np.sum(res))

    # for the final one, we will add replicas of the final example.
    count, fakes = _process_final((args.num_folds-1, args, total))
    total += count
    print('{} final examplds written, which includes {} fakes to make batch size work.'.format(total, fakes))
    
    
if __name__ == '__main__':
    main()
