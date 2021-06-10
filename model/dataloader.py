"""
Bidirectional dataloader
"""
import sys

sys.path.append('../')
import time
from copy import deepcopy

import tensorflow as tf
from utils.encode.encoder import START, NEXTCAPTION_START
from utils.model_utils import get_shape_list, encode_string, lightweight_image_augment, sample_bernoulli, resize_and_pad, pad_to_fixed_size
from utils.neat_config import NeatConfig

slim_example_decoder = tf.contrib.slim.tfexample_decoder

###################################
# Data loading stuff v3
chunk_k2f = {
    'image/encoded': tf.io.FixedLenFeature((), tf.string, default_value=''),
    'image/format': tf.io.FixedLenFeature((), tf.string, default_value='jpeg'),
    'image/key/sha256': tf.io.FixedLenFeature((), tf.string, default_value=''),
    'image/height': tf.io.FixedLenFeature((), tf.int64, 1),
    'image/width': tf.io.FixedLenFeature((), tf.int64, 1),
    'youtube_id': tf.io.FixedLenFeature((), tf.string, default_value=''),
    'tokenized_cleaned_asr': tf.io.VarLenFeature(tf.int64),
    'tokenized_raw_asr': tf.io.VarLenFeature(tf.int64),
    'is_eoc': tf.io.FixedLenFeature((), tf.int64, 1),
    'mean_time': tf.io.FixedLenFeature((), tf.float32, 1),
    'chunk_num': tf.io.FixedLenFeature((), tf.int64, 1),
}

def _decode_record(record, NUM_CHUNKS):
    keys_to_features = {f'c{i:02d}/{k}': v for i in range(NUM_CHUNKS) for k, v in chunk_k2f.items()}
    items_to_handlers = {k: (slim_example_decoder.Tensor(k)) for k in keys_to_features.keys()}

    serialized_example = tf.reshape(record, shape=[])
    decoder = slim_example_decoder.TFExampleDecoder(keys_to_features,
                                                    items_to_handlers)
    keys = sorted(decoder.list_items())
    tensors = decoder.decode(serialized_example, items=keys)

    # Turn everything into chunks, make it reasonable
    tensor_dict = dict(zip(keys, tensors))

    # Turn into a list of chunks
    chunk_list = []
    for i in range(NUM_CHUNKS):
        cur_chunk = {}
        for k in sorted(chunk_k2f.keys()):
            cur_chunk[k] = tensor_dict.pop(f'c{i:02d}/{k}')
        chunk_list.append(cur_chunk)
    return chunk_list


def _dataset_parser(value, config, is_training):
    DESIRED_OUTPUT_SIZE = config['image_size']
    NUM_CHUNKS = config['num_chunks']

    with tf.name_scope('parser'):

        # We got a list of chunks here
        chunk_list = _decode_record(value, NUM_CHUNKS=NUM_CHUNKS)

        features = {
            'youtube_id': tf.stack([encode_string(v['youtube_id'], 64) for v in chunk_list], 0),
            'chunk_num': tf.cast(tf.stack([v['chunk_num'] for v in chunk_list], 0), dtype=tf.int32),
            'mean_time': tf.stack([v['mean_time'] for v in chunk_list], 0),
        }

        # tf.map_fn avoids memory usage
        def _load_and_resize_img(x):
            img = tf.image.decode_jpeg(x, channels=3)
            img = tf.image.convert_image_dtype(img, dtype=tf.float32)

            # Maybe vary the resize method, there are 4 in total
            img, this_image_info = resize_and_pad(img, DESIRED_OUTPUT_SIZE,
                                                  do_random_scale=True,
                                                  random_scale_max=config.get('random_scale_max', 1.05),
                                                  random_scale_min=config.get('random_scale_min', 0.95),
                                                  resize_method='random')
            # occasionally a NaN sneaks in here for conceptual captions
            img = tf.where(tf.math.is_finite(img), img, tf.zeros_like(img))

            if config.get('augment_prob', 0.0) > 0.0:
                img = lightweight_image_augment(img,
                                                augment_prob=config['augment_prob'],
                                                allowed_transforms='brightness,contrast')
            return img

        encodeds = tf.stack([x['image/encoded'] for x in chunk_list])
        images = tf.map_fn(_load_and_resize_img, elems=encodeds, dtype=tf.float32)

        if config.get('use_bfloat16', False):
            tf.logging.info("image -> bfloat")
            images = tf.cast(images, dtype=tf.bfloat16)

        features['images'] = images

        # Handle all text
        do_clean_prob = config.get('clean_asr_prob', 0.5)
        do_clean = sample_bernoulli(do_clean_prob)
        len_per_chunk = config.get('chunk_text_len', 32)  # This is INCLUDING things like [THISCAPTIO] and [NEXTCAPTION]
        tf.logging.info("Doing clean with prob={:.3f}. len per chunk = {}".format(do_clean_prob, len_per_chunk))

        # ASR captions and stuff
        # I think using ragged might be faster idk
        asr_caption = tf.cond(do_clean,
                              true_fn=lambda: tf.ragged.stack([c['tokenized_cleaned_asr'][None] for c in chunk_list]),
                              false_fn=lambda: tf.ragged.stack([c['tokenized_raw_asr'][None] for c in chunk_list])
                              )
        # Undo raggedness
        asr_caption = tf.squeeze(asr_caption.to_tensor(default_value=0), 1)
        asr_caption = tf.cast(asr_caption, dtype=tf.int32)

        # Add the start token
        start_token = tf.where(do_clean, START, NEXTCAPTION_START)
        asr_caption = tf.concat([tf.fill(dims=[NUM_CHUNKS, 1], value=start_token), asr_caption], 1)
        features['input_ids'] = pad_to_fixed_size(asr_caption, pad_value=0, output_shape=[NUM_CHUNKS, len_per_chunk],
                                                  truncate=True, axis=1)

        # Last segment is always "end"
        features['is_eoc'] = tf.cast(tf.stack([c['is_eoc'] for c in chunk_list[:-1]] + [1]), dtype=tf.bool)
        chunk_id_delta = tf.concat([[0], tf.cast(features['is_eoc'][:-1], dtype=tf.int32)], 0)
        features['video_src_ids'] = tf.cumsum(chunk_id_delta)
        return features, {}


def input_fn_builder(config: NeatConfig, is_training, dataset_parser=_dataset_parser):
    """
    :param config: NeatConfig object containing model/data

    :param is_training:
    :return:
    """
    merged_config = deepcopy(config.data)
    merged_config.update(deepcopy(config.model))
    input_file = config.data['train_file'] if is_training else config.data['val_file']
    num_files = len(tf.gfile.Glob(input_file))
    num_threads = config.data.get('num_threads', 64)

    def input_fn(params):
        # this is a reserved term
        batch_size = params['batch_size']
        if 'context' in params:
            current_host = params['context'].current_input_fn_deployment()[1]
            num_hosts = params['context'].num_hosts
        else:
            current_host = 0
            num_hosts = 1
        tf.logging.info(f"Current host = {current_host} num hosts = {num_hosts}. Local batch size = {batch_size}")

        if num_hosts == 1:
            tf.logging.info("Shuffling files and NOT sharding the data across workers!")
            dataset = tf.data.Dataset.list_files(
                input_file, shuffle=is_training,
                seed=tf.compat.v1.random.set_random_seed(int(time.time() * 1e9)))

            # `cycle_length` is the number of parallel files that get read.
            cycle_length = min(num_threads, num_files)
        else:
            # For multi-host training, we want each hosts to always process the same
            # subset of files.  Each host only sees a subset of the entire dataset
            assert (num_files // num_hosts) >= 1
            dataset = tf.data.Dataset.list_files(input_file, shuffle=False)
            dataset = dataset.shard(num_shards=num_hosts, index=current_host)
            cycle_length = min(num_threads, num_files // num_hosts)

        tf.logging.info(f"num_files={num_files}, num_threads={num_threads} cycle_length={cycle_length}")

        if is_training:
            dataset = dataset.repeat()

        dataset = dataset.apply(
            tf.data.experimental.parallel_interleave(
                lambda file_name: tf.data.TFRecordDataset(file_name).prefetch(1),
                cycle_length=cycle_length,
                sloppy=is_training))

        if is_training:
            # NOTE: I'm scaling down the buffer size
            buffer_size = config.data.get('shuffle_buffer_size', 256)
            tf.logging.info(f"Shuffle buffer size of {buffer_size}! You might need to tune this for smaller datasets")
            dataset = dataset.shuffle(buffer_size=buffer_size)

        dataset = dataset.map(lambda x: dataset_parser(x, merged_config, is_training=is_training),
                              num_parallel_calls=num_threads)
        dataset = dataset.batch(batch_size, drop_remainder=True)

        # This is in case 16 is too large of a size, so we would need to use a smaller number of frames
        unbatch_size = config.data.get('unbatch_size', 1)
        if unbatch_size > 1:
            assert batch_size == 1
            def _unbatch(features, labels):
                for k in sorted(features.keys()):
                    sl = get_shape_list(features[k])
                    if sl[0] != 1:
                        raise ValueError(f"{k} not bsize=1?")
                    assert sl[0] == 1
                    assert sl[1] == merged_config['num_chunks']
                    new_sl = [unbatch_size, 1, merged_config['num_chunks'] // unbatch_size] + sl[2:]
                    tf.logging.info(f"Unbatching unbatch_size={unbatch_size}: {k} {sl} -> {new_sl}")

                    features[k] = tf.reshape(features[k], new_sl)
                return features, labels

            dataset = dataset.map(_unbatch, num_parallel_calls=1)
            dataset = dataset.unbatch()

        def _process_example(features, labels):
            if config.data.get('shuffle_chunks', False):
                # basically when I made the tfrecords I often had stuff like [A A A A B B] and we want to
                # randomly make it like [B B A A A A]
                tf.logging.info("Shuffling chunks in features")

                bsz, nchunk = get_shape_list(features['video_src_ids'])
                chunkid_to_new_id_mapping = tf.argsort(tf.random_uniform([bsz, nchunk]), -1)
                new_chunkid = tf.gather(chunkid_to_new_id_mapping, features['video_src_ids'], batch_dims=1)
                trg_idx = new_chunkid * nchunk + tf.range(nchunk, dtype=tf.int32)[None]
                idx = tf.argsort(trg_idx, 1)
                for k in ['youtube_id', 'chunk_num', 'mean_time', 'images', 'input_ids', 'is_eoc', 'video_src_ids']:
                    features[k] = tf.gather(features[k], idx, batch_dims=1)

            # Shuffle the frames
            shuffle_prob = merged_config.get('image_shuffle_prob', 0.5)
            shuffle_offset = 16

            num_chunks_in_group = merged_config['num_chunks_in_group']
            batch_size_, num_chunks, l_ = get_shape_list(features['input_ids'])
            B = batch_size_ * num_chunks // num_chunks_in_group

            if shuffle_prob < 1e-6:
                tf.logging.info(f"NOT shuffling img")
                features['shuffled_idx_img'] = tf.reshape(tf.tile(tf.range(num_chunks_in_group)[None], [B, 1]),
                                                           [-1])
            else:
                tf.logging.info("Shuffling  chunks with probability {:.3f}".format(shuffle_prob))
                # With probability shuffle_prob, we will shuffle the chunks
                # If we do shuffle then sample uniformly
                # Skip shuffling just one thing -- that's the same as shuffling nothing
                num_shuffle_per_group_probs = [1.0 - shuffle_prob, 1e-6] + [
                    shuffle_prob / (num_chunks_in_group - 1) for i in range(num_chunks_in_group - 1)]
                ev = sum([i * p for i, p in enumerate(num_shuffle_per_group_probs)])
                tf.logging.info(
                    "probs: {}\nExpected # of {}s out of place: {:.3f}".format(num_shuffle_per_group_probs, k, ev))
                nspg_logprob = tf.math.log(num_shuffle_per_group_probs)[None]

                #################################
                num_shuffle_img = tf.squeeze(tf.random.categorical(nspg_logprob, dtype=tf.int32, num_samples=B), 0)
                do_shuffle_img = tf.less(tf.argsort(tf.random_uniform([B, num_chunks_in_group]), 1),
                                       num_shuffle_img[:, None])
                shuffled_idx_img = tf.where(
                    do_shuffle_img,
                    shuffle_offset + tf.argsort(tf.random_uniform([B, num_chunks_in_group]), 1),
                    tf.tile(tf.range(num_chunks_in_group)[None], [B, 1]),
                )
                features['shuffled_idx_img'] = tf.reshape(shuffled_idx_img, [-1])

            # Flatten bsize and transpose
            batch_size_, num_imgs_per_batch, h, w, three_ = get_shape_list(features['images'], 5)
            features['images'] = tf.reshape(features['images'], [batch_size_ * num_imgs_per_batch, h, w, 3])
            if is_training and config.model.get('transpose_input', False):
                # TPUs don't handle shapes with small numbers like "3" very well so we do this thing
                features['images'] = tf.transpose(features['images'], [1, 2, 3, 0])

            tf.logging.info("features~")
            for k, v in features.items():
                tf.logging.info("{}: {}".format(k, v.shape))
            tf.logging.info("labels~")
            for k, v in labels.items():
                tf.logging.info("{}: {}".format(k, v.shape))
            return features, labels

        # Batch level stuff
        if is_training:
            dataset = dataset.map(_process_example, num_parallel_calls=num_threads)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

    return input_fn
