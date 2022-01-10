"""
Turn the merged data into some tfrecord files.
"""
import argparse
import hashlib
import io
import json
import os
import random
import numpy as np
from tempfile import TemporaryDirectory
from copy import deepcopy

import PIL.Image
import tensorflow as tf
from google.cloud import storage
from utils.encode.encoder import get_encoder
from sacremoses import MosesDetokenizer
import regex as re
from tqdm import tqdm

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = '../xlabprojects-keys.json'

parser = argparse.ArgumentParser(description='SCRAPE!')
parser.add_argument(
    '-fold',
    dest='fold',
    default=0,
    type=int,
    help='which fold we are on'
)
parser.add_argument(
    '-num_folds',
    dest='num_folds',
    default=1,
    type=int,
    help='Number of folds (corresponding to both the number of training files and the number of testing files)',
)
parser.add_argument(
    '-seed',
    dest='seed',
    default=1337,
    type=int,
    help='which seed to use'
)
parser.add_argument(
    '-base_fn',
    dest='base_fn',
    default='gs://tubert_euro/data/vcr_frist_round/train',
    type=str,
    help='Base filename to use. You can start this with gs:// and we\'ll put it on google cloud.'
)

parser.add_argument(
    '-image_dir',
    dest='image_dir',
    default='/home/lux32/vlmodel/data/vcr/vcr1images/',
    type=str,
    help='Image directory.'
)

parser.add_argument(
    '-annotations_file',
    dest='annotations_file',
    default='/home/lux32/vlmodel/data/vcr/annotation/train.jsonl',
    type=str,
    help='Question annotations.'
)

parser.add_argument(
    '-max_seq_length',
    dest='max_seq_length',
    default=512,
    type=int,
    help='Max sequence length',
)

parser.add_argument(
    '-ans_num',
    dest='ans_num',
    default=-1,
    type=int,
    help='answer for q->a',
)

parser.add_argument(
    '-rat_num',
    dest='rat_num',
    default=-1,
    type=int,
    help='answer for qa->r',
)

args = parser.parse_args()
random.seed(args.seed + args.fold)

MAX_NUM_OBJS = 84


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


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def create_tf_example(item,
                      image_dir):
    """Converts image and annotations to a tf.Example proto.

    Args:
      image: dict with keys:
        [u'license', u'file_name', u'coco_url', u'height', u'width',
        u'date_captured', u'flickr_url', u'id']
      image_dir: directory containing the image files.
    Returns:
      example: The converted tf.Example
      num_annotations_skipped: Number of (invalid) annotations that were ignored.

    Raises:
      ValueError: if the image pointed to by data['filename'] is not a valid JPEG
    """
    num_annotations_skipped = 0

    feature_dict = {
        'img_id':
            bytes_feature(item['img_id'].encode('utf8')),
        'annot_id':
            bytes_feature(item['annot_id'].encode('utf8')),
    }

    for mode in ['answer', 'rationale']:

        for draw in ['segm', 'bbox']:
            split = item["annot_id"].split("-")[0]
            annot_img_path = f'{draw}/{split}/{mode}/{item["annot_id"]}.jpg'

            if not os.path.isfile(annot_img_path):
                num_annotations_skipped += 1
                return None, None, num_annotations_skipped

            with tf.gfile.GFile(annot_img_path, 'rb') as fid:
                encoded_jpg = fid.read()
            key = hashlib.sha256(encoded_jpg).hexdigest()

            feature_dict.update({
                f'{mode}/{draw}/image/key/sha256':
                    bytes_feature(key.encode('utf8')),
                f'{mode}/{draw}/image/encoded':
                    bytes_feature(encoded_jpg),
                f'{mode}/{draw}/image/format':
                    bytes_feature('jpeg'.encode('utf8')),
            })

        feature_dict.update({
            f'{mode}/ctx':
                int64_list_feature(item[f'{mode}_ctx']),
            f'{mode}/ctx_tags':
                int64_list_feature(item[f'{mode}_ctx_tags']),
        })

        for i in range(4):
            feature_dict.update({
                f'{mode}/choice_{i}':
                    int64_list_feature(item[f'{mode}_choice_{i}']),
                f'{mode}/choice_tags_{i}':
                    int64_list_feature(item[f'{mode}_choice_tags_{i}']),
            })

    feature_dict.update({
        'label/answer_label':
            int64_feature(item['answer_label']),
        'label/rationale_label':
            int64_feature(item['rationale_label']),
    })

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return key, example, num_annotations_skipped


GENDER_NEUTRAL_NAMES = ['Casey', 'Riley', 'Jessie', 'Jackie', 'Avery', 'Jaime', 'Peyton', 'Kerry', 'Jody', 'Kendall',
                        'Skyler', 'Frankie', 'Pat', 'Quinn', 'Morgan', 'Finley', 'Harley', 'Robbie', 'Sidney', 'Tommie',
                        'Ashley', 'Carter', 'Adrian', 'Clarke', 'Logan', 'Mickey', 'Nicky', 'Parker', 'Tyler',
                        'Reese', 'Charlie', 'Austin', 'Denver', 'Emerson', 'Tatum', 'Dallas', 'Haven', 'Jordan',
                        'Robin', 'Rory', 'Bellamy', 'Salem', 'Sutton', 'Gray', 'Shae', 'Kyle', 'Alex', 'Ryan',
                        'Cameron', 'Dakota']


class VCRDataset:
    '''
    Here's an example jsonl
    {
    "movie": "3015_CHARLIE_ST_CLOUD",
    "objects": ["person", "person", "person", "car"],
    "interesting_scores": [0],
    "answer_likelihood": "possible",
    "img_fn": "lsmdc_3015_CHARLIE_ST_CLOUD/3015_CHARLIE_ST_CLOUD_00.23.57.935-00.24.00.783@0.jpg",
    "metadata_fn": "lsmdc_3015_CHARLIE_ST_CLOUD/3015_CHARLIE_ST_CLOUD_00.23.57.935-00.24.00.783@0.json",
    "answer_orig": "No she does not",
    "question_orig": "Does 3 feel comfortable?",
    "rationale_orig": "She is standing with her arms crossed and looks disturbed",
    "question": ["Does", [2], "feel", "comfortable", "?"],
    "answer_match_iter": [3, 0, 2, 1],
    "answer_sources": [3287, 0, 10184, 2260],
    "answer_choices": [
        ["Yes", "because", "the", "person", "sitting", "next", "to", "her", "is", "smiling", "."],
        ["No", "she", "does", "not", "."],
        ["Yes", ",", "she", "is", "wearing", "something", "with", "thin", "straps", "."],
        ["Yes", ",", "she", "is", "cold", "."]],
    "answer_label": 1,
    "rationale_choices": [
        ["There", "is", "snow", "on", "the", "ground", ",", "and",
            "she", "is", "wearing", "a", "coat", "and", "hate", "."],
        ["She", "is", "standing", "with", "her", "arms", "crossed", "and", "looks", "disturbed", "."],
        ["She", "is", "sitting", "very", "rigidly", "and", "tensely", "on", "the", "edge", "of", "the",
            "bed", ".", "her", "posture", "is", "not", "relaxed", "and", "her", "face", "looks", "serious", "."],
        [[2], "is", "laying", "in", "bed", "but", "not", "sleeping", ".",
            "she", "looks", "sad", "and", "is", "curled", "into", "a", "ball", "."]],
    "rationale_sources": [1921, 0, 9750, 25743],
    "rationale_match_iter": [3, 0, 2, 1],
    "rationale_label": 1,
    "img_id": "train-0",
    "question_number": 0,
    "annot_id": "train-0",
    "match_fold": "train-0",
    "match_index": 0,
    }
    '''
    def __init__(self, annotations: str):
        with tf.gfile.GFile(annotations, 'r') as fid:
            self.items = [json.loads(s) for s in fid]
        self.encoder = get_encoder()
        self.detokenizer = MosesDetokenizer(lang='en')
        self.obj_pat = re.compile("OBJ-[0-9]+")

    def get_tokenization_with_tags(self, text, objects, pad_ide=0):
        """Tokenize text and associate object tag with each token.

        Args:
            text: A list contains original tokenization in item.
            objects: A list contains original objects in item.
            pad_ide: tag id for token without object

        Returns:
            token_ides: a list contains bpe token ids of the text
            obj_tags: a list contains object tags for each bpe token, same length with token_ides
        """
        moses_tokens, obj_lists, obj_index = [], [], 0
        for word_token in text:
            if isinstance(word_token, list):
                obj_lists.append(word_token)
                moses_tokens.append(f'OBJ-{obj_index}')
                obj_index += 1
            else:
                moses_tokens.append(word_token)
        sentence = self.detokenizer.detokenize(moses_tokens)

        # If we add the image as an extra box then the 0th will be the image.
        obj_pos_ids = np.arange(len(objects), dtype=np.int32) + 1

        token_ides, obj_tags = [], []
        for i, word_token in enumerate(sentence.split()):
            is_obj = self.obj_pat.search(word_token)
            if is_obj:
                start_idx, end_idx = is_obj.span()
                pre, post = word_token[:start_idx], word_token[end_idx:]
                if pre:
                    tokenization = self.encoder.encode(f' {pre}' if i else pre)
                    token_ides.extend(tokenization)
                    obj_tags.extend([pad_ide] * len(tokenization))

                obj_list = obj_lists[int(word_token[start_idx:end_idx].split('-')[-1])]
                for j, object_id in enumerate(obj_list):
                    if 1 < len(obj_list) == j + 1:
                        tokenization = self.encoder.encode(' and')
                        token_ides.extend(tokenization)
                        obj_tags.extend([pad_ide] * len(tokenization))
                    obj = objects[object_id]
                    obj = GENDER_NEUTRAL_NAMES[object_id % len(GENDER_NEUTRAL_NAMES)] if obj == 'person' else obj
                    obj_word = obj if start_idx else f' {obj}' if i + j else obj.capitalize()
                    obj_word_tokenization = self.encoder.encode(obj_word)
                    token_ides.extend(obj_word_tokenization)
                    obj_tags.extend([obj_pos_ids[object_id]] * len(obj_word_tokenization))

                if post:
                    tokenization = self.encoder.encode(post)
                    token_ides.extend(tokenization)
                    obj_tags.extend([pad_ide] * len(tokenization))
            else:
                word_tokenization = self.encoder.encode(f' {word_token}' if i else word_token)
                token_ides.extend(word_tokenization)
                obj_tags.extend([pad_ide] * len(word_tokenization))

        assert len(token_ides) == len(obj_tags), 'each token much correspond with a tag'

        return token_ides, obj_tags

    def process(self, input_item):
        """Process item and extract feature.

        Args:
            input_item: An item from original jsonl as a dictionary.
        Returns:
            new_item: A dictionary contains extracted feature for the item
        """
        with open(os.path.join(args.image_dir, input_item['metadata_fn']), 'r') as f:
            metadata = json.load(f)

        if 'answer_label' not in input_item:
            input_item['answer_label'] = args.ans_num
        if 'rationale_label' not in input_item:
            input_item['rationale_label'] = args.rat_num

        new_item = {
            'img_id': input_item['img_id'],
            'annot_id': input_item['annot_id'],
            'img_fn': input_item['img_fn'],
            'answer_label': input_item['answer_label'],
            'rationale_label': input_item['rationale_label']
        }

        image = PIL.Image.open(os.path.join(args.image_dir, input_item['img_fn']))
        w, h = image.size

        for mode in ['answer', 'rationale']:
            item = deepcopy(input_item)

            ctx = item['question']
            if mode == 'rationale':
                ctx += item['answer_choices'][item['answer_label']]

            ctx_ids, ctx_tags = self.get_tokenization_with_tags(ctx, item['objects'])
            new_item.update({f'{mode}_ctx': ctx_ids,
                             f'{mode}_ctx_tags': ctx_tags})

            for i, choice in enumerate(item[f'{mode}_choices']):
                ans_ids, ans_tags = self.get_tokenization_with_tags(choice, item['objects'])
                new_item.update({f'{mode}_choice_{i}': ans_ids,
                                 f'{mode}_choice_tags_{i}': ans_tags})

        return new_item


out_file = '{}-{:05d}-of-{:05d}.tfrecord'.format(args.base_fn, args.fold, args.num_folds)

vcr_data = VCRDataset(args.annotations_file)
print("LOAD VCR ANNOTATIONS", flush=True)

total_num_questions_skipped = 0
num_written = 0
with GCSTFRecordWriter(out_file) as tfrecord_writer:
    for idx, item in enumerate(tqdm(vcr_data.items)):

        if idx % args.num_folds != args.fold:
            continue
        if (idx // args.num_folds) % 100 == 0:
            tf.logging.info('On image %d of %d', idx, len(vcr_data.items) // args.num_folds)

        _, tf_example, num_questions_skipped = create_tf_example(vcr_data.process(item), image_dir=args.image_dir)

        if num_questions_skipped > 0:
            total_num_questions_skipped += num_questions_skipped
        else:
            tfrecord_writer.write(tf_example.SerializeToString())
            num_written += 1

print(f'Finished writing {num_written} questions, skipped {total_num_questions_skipped} total', flush=True)