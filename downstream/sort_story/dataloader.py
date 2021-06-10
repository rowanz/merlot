import os
import time
import pprint
from copy import deepcopy

import tensorflow as tf
from utils.encode.encoder import PADDING, START, END, NEXTCAPTION_START
from utils.model_utils import encode_string, get_shape_list, resize_and_pad, pad_to_fixed_size
from utils.encode.encoder import get_encoder

slim_example_decoder = tf.contrib.slim.tfexample_decoder

###################################
# Data loading stuff v2
def _process_keys_to_features(config):
    keys_to_features = {
        'permutation_identity_encode': tf.io.FixedLenFeature((), tf.int64, default_value=-1),
        'story_id': tf.io.FixedLenFeature((), tf.int64, default_value=-1),
    }

    for i in range(config['num_chunks']):
        keys_to_features.update({'sentence/sentence_{}'.format(i) : tf.io.VarLenFeature(tf.int64),})

    for i in range(config['num_chunks']):
        keys_to_features.update({
            'image/image_{}_sha256'.format(i): tf.io.FixedLenFeature((), tf.string, default_value=''),
            'image/image_{}_encoded'.format(i): tf.io.FixedLenFeature((), tf.string, default_value=''),
            'image/image_{}_format'.format(i): tf.io.FixedLenFeature((), tf.string, default_value='jpeg'),
        })

    return keys_to_features


def _process_items_to_handlers(config):
    items_to_handlers = {
        'permutation_identity_encode': (slim_example_decoder.Tensor('permutation_identity_encode')),
        'story_id': (slim_example_decoder.Tensor('story_id')),
    }

    for i in range(config['num_chunks']):
        items_to_handlers.update({f'sentence_{i}' : (slim_example_decoder.Tensor(f'sentence/sentence_{i}'))})

    for i in range(config['num_chunks']):
        items_to_handlers.update({
            f'image_{i}':
                slim_example_decoder.Image(
                    image_key=f'image/image_{i}_encoded',
                    format_key=f'image/image_{i}_format',
                    channels=3),
            f'image_{i}_key': (slim_example_decoder.Tensor(f'image/image_{i}_sha256')),
        })

    return items_to_handlers


def _decode_record(record, config):
    """Decodes serialized tensorflow example and returns a tensor dictionary.
    """
    serialized_example = tf.reshape(record, shape=[])
    keys_to_features, items_to_handlers = _process_keys_to_features(config), _process_items_to_handlers(config)

    decoder = slim_example_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)
    keys = sorted(decoder.list_items())

    tensors = decoder.decode(serialized_example, items=keys)
    tensor_dict = dict(zip(keys, tensors))
    for i in range(config['num_chunks']):
        tensor_dict[f'image_{i}'].set_shape([None, None, 3])
    return tensor_dict


def _dataset_parser(value, config, is_training):
    """Parse data to a fixed dimension input image and learning targets.
    """
    MAX_NUM_TOKENS = 32

    with tf.name_scope('parser'):
        data = _decode_record(value, config)

        # We'll build this one up
        
        features = {'permutation_identity_encode': data['permutation_identity_encode'],
                    'story_id': data['story_id']}

        for i in range(config['num_chunks']):
            # Normalize img
            image = tf.image.convert_image_dtype(data[f'image_{i}'], dtype=tf.float32)
            image, image_info = resize_and_pad(image, config['image_size'], do_random_scale=False)

            # bfloat16 and commit the image
            if config.get('use_bfloat16', False):
                tf.logging.info("image -> bfloat")
                image = tf.cast(image, dtype=tf.bfloat16)
            data[f'image_{i}'], data[f'image_info_{i}'] = image, image_info

        features['images'] = tf.stack([data[f'image_{i}'] for i in range(config['num_chunks'])])
        features['image_info'] = tf.stack([data[f'image_info_{i}'] for i in range(config['num_chunks'])])

        for i in range(config['num_chunks']):
            data[f'sentence_{i}'] = tf.concat([tf.fill([1], START),
                                               tf.cast(data[f'sentence_{i}'], tf.int32)], 0)

        features['sentences'] = tf.stack([pad_to_fixed_size(data[f'sentence_{i}'], pad_value=PADDING, output_shape=[MAX_NUM_TOKENS]) for i in range(config['num_chunks'])])

        ###########
        labels = {}
        features.update(labels)

        return features, labels


BUFFER_SIZE = None
def input_fn_builder(config, is_training):
    """
    :param config: NeatConfig object containing model/data

    :param is_training:
    :return:
    """
    merged_config = deepcopy(config.data)
    merged_config.update(deepcopy(config.model))
    merged_config.update(deepcopy(config.downstream))
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
            # If pretraining on a v3-512, there are 128 files per
            # each one has 128 examples so there are 16384 examples
            # Probably a good idea to make the shuffle size be a sizeable fraction. i'm saying 1/8 by default
            # since there will also be considerable shuffling of the files too
            buffer_size = config.data.get('shuffle_buffer_size', 64)
            tf.logging.info(f"Shuffle buffer size of {buffer_size}! You might need to tune this for smaller datasets")
            dataset = dataset.shuffle(buffer_size=buffer_size)
        dataset = dataset.map(lambda x: _dataset_parser(x, merged_config, is_training=is_training),
                              num_parallel_calls=num_threads)
        dataset = dataset.batch(batch_size, drop_remainder=True)

        def _process_example(features, labels):
            if is_training:
                with tf.name_scope('flatten_examples'):
                    # Flatten bsize and transpose
                    batch_size_, num_imgs_per_batch, h, w, three_ = get_shape_list(features['images'], 5)
                    features['images'] = tf.reshape(features['images'], [batch_size_ * num_imgs_per_batch, h, w, 3])

            if is_training and config.model['transpose_input']:
                # TPUs don't handle shapes with small numbers like "3" very well so we do this thing
                features['images'] = tf.transpose(features['images'], [1, 2, 3, 0])

            return features, labels

        dataset = dataset.map(_process_example, num_parallel_calls=64)
        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

        return dataset

    return input_fn
