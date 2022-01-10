# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Data loader and processing.
   VQA tf-record using old bpe tokenizer, so don't forget to add 100 to each token
"""
import os
import time
from copy import deepcopy

import tensorflow as tf
from utils.encode.encoder import PADDING, START, END
from utils.model_utils import encode_string, get_shape_list, resize_and_pad, pad_to_fixed_size
from utils.encode.encoder import get_encoder


encoder = get_encoder()
slim_example_decoder = tf.contrib.slim.tfexample_decoder

VCR_prompt = {
    'answer': [START] + encoder.encode(' answer question:'),
    'rationale': [START] + encoder.encode(' provide rationale:'),
}


###################################
# Data loading stuff v2

def _process_keys_to_features(config):
    mode = config['mode']

    keys_to_features = {
        'img_id': tf.io.FixedLenFeature((), tf.string, default_value=''),
        'annot_id': tf.io.FixedLenFeature((), tf.string, default_value=''),

        f'{mode}/{config["draw"]}/image/encoded': tf.io.FixedLenFeature((), tf.string, default_value=''),
        f'{mode}/{config["draw"]}/image/format': tf.io.FixedLenFeature((), tf.string, default_value='jpeg'),
        f'{mode}/{config["draw"]}/image/key/sha256': tf.io.FixedLenFeature((), tf.string, default_value=''),
    }

    keys_to_features.update({
        f'label/{mode}_label': tf.io.FixedLenFeature((), tf.int64, -1),
        f'{mode}/ctx': tf.io.VarLenFeature(tf.int64),
        f'{mode}/ctx_tags': tf.io.VarLenFeature(tf.int64),
    })


    for i in range(4):
        keys_to_features.update({
            f'{mode}/choice_{i}': tf.io.VarLenFeature(tf.int64),
            f'{mode}/choice_tags_{i}': tf.io.VarLenFeature(tf.int64),
        })

    return keys_to_features


def _process_items_to_handlers(config):
    mode = config['mode']

    items_to_handlers = {
        'img_id': (slim_example_decoder.Tensor('img_id')),
        'annot_id': (slim_example_decoder.Tensor('annot_id')),

        'image':
            slim_example_decoder.Image(
                image_key=f'{mode}/{config["draw"]}/image/encoded',
                format_key=f'{mode}/{config["draw"]}/image/format',
                channels=3),
        'key': (slim_example_decoder.Tensor(f'{mode}/{config["draw"]}/image/key/sha256')),
    }

    items_to_handlers.update({
        'label': (slim_example_decoder.Tensor(f'label/{mode}_label')),
        'ctx': (slim_example_decoder.Tensor(f'{mode}/ctx')),
        'ctx_tags': (slim_example_decoder.Tensor(f'{mode}/ctx_tags')),
    })

    for i in range(4):
        items_to_handlers.update({
            f'choice_{i}': (slim_example_decoder.Tensor(f'{mode}/choice_{i}')),
            f'choice_tags_{i}': (slim_example_decoder.Tensor(f'{mode}/choice_tags_{i}')),
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
    tensor_dict['image'].set_shape([None, None, 3])
    return tensor_dict


def _dataset_parser(value, config, is_training):
    """Parse data to a fixed dimension input image and learning targets.
    """
    MAX_NUM_TOKENS = 134 if config['mode'] == 'answer' else 184

    with tf.name_scope('parser'):
        data = _decode_record(value, config)

        # We'll build this one up
        features = {
            'img_id': encode_string(data['img_id'], 24),
            'annot_id': encode_string(data['annot_id'], 24),
        }

        # Normalize img
        image = tf.image.convert_image_dtype(data['image'], dtype=tf.float32)
        image, features['image_info'], _ = \
            resize_and_pad(image, config['image_size'], do_random_scale=False,
                           random_scale_max=1.1, random_scale_min=0.9)

        # bfloat16 and commit the image
        if config.get('use_bfloat16', False):
            tf.logging.info("image -> bfloat")
            image = tf.cast(image, dtype=tf.bfloat16)
        features['images'] = image

        for i in range(4):
            data[f'ctx_choice_{i}'] = tf.concat([tf.constant(VCR_prompt[config['mode']], dtype=tf.int32),
                                                 tf.cast(data['ctx'], tf.int32),
                                                 tf.cast(data[f'choice_{i}'], tf.int32),
                                                 tf.fill([1], END)], axis=0)

        lm_input = tf.stack([pad_to_fixed_size(data[f'ctx_choice_{i}'],
                                               pad_value=PADDING, output_shape=[MAX_NUM_TOKENS]) for i in range(4)])
        lm_mask = tf.cast(tf.not_equal(lm_input, PADDING), tf.int32)

        features['lm_input'], features['lm_mask'] = lm_input, lm_mask

        ###########
        labels = {
            'lm_targets': data['label'],
        }
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

    def input_fn(params):
        batch_size = params['batch_size']
        dataset = tf.data.Dataset.list_files(input_file, shuffle=is_training,
                                             seed=tf.random.set_random_seed(int(time.time() * 1e9)))
        if is_training:
            dataset = dataset.repeat()

        # Prefetch data from files.
        def _prefetch_dataset(filename):
            data = tf.data.TFRecordDataset(filename, buffer_size=BUFFER_SIZE).prefetch(1)
            return data

        dataset = dataset.apply(tf.contrib.data.parallel_interleave(_prefetch_dataset, cycle_length=32,
                                                                    sloppy=is_training))

        if is_training:
            dataset = dataset.shuffle(buffer_size=64)

        dataset = dataset.map(lambda x: _dataset_parser(x, merged_config, is_training=is_training), num_parallel_calls=64)
        dataset = dataset.batch(batch_size, drop_remainder=True)

        def _process_example(features, labels):
            if is_training and config.model['transpose_input']:
                # TPUs don't handle shapes with small numbers like "3" very well so we do this thing
                features['images'] = tf.transpose(features['images'], [1, 2, 3, 0])

            if is_training:
                # Turn [batch_size, 4, ...] things into [batch_size * 4]
                with tf.name_scope('flatten_examples'):
                    for k in ['lm_input', 'lm_mask']:
                        features[k] = tf.reshape(features[k], [batch_size * 4] + get_shape_list(features[k])[2:])
            return features, labels

        dataset = dataset.map(_process_example, num_parallel_calls=64)
        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

        return dataset

    return input_fn


if __name__ == '__main__':
    # Test the dataloader
    from utils.neat_config import NeatConfig

    use_eager = False
    if use_eager:
        tf.compat.v1.enable_eager_execution()

    use_train = False
    mode = tf.estimator.ModeKeys.TRAIN if use_train else tf.estimator.ModeKeys.PREDICT

    config = NeatConfig.from_args("Pretraining script", default_config_file='../../model/configs/merlot_vcr.yaml')

    train_input_fn = input_fn_builder(config, is_training=use_train)

    if not use_eager:
        next_item = train_input_fn(params={'batch_size': 8}).make_one_shot_iterator().get_next()
        var_init = tf.global_variables_initializer()
        table_init = tf.tables_initializer()
        sess = tf.Session()
        features, labels = sess.run(next_item)

        import ipdb
        ipdb.set_trace()
