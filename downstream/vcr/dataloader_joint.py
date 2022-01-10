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
    keys_to_features = {
        'img_id': tf.io.FixedLenFeature((), tf.string, default_value=''),
        'annot_id': tf.io.FixedLenFeature((), tf.string, default_value=''),

        f'answer/{config["draw"]}/image/encoded': tf.io.FixedLenFeature((), tf.string, default_value=''),
        f'answer/{config["draw"]}/image/format': tf.io.FixedLenFeature((), tf.string, default_value='jpeg'),
        f'answer/{config["draw"]}/image/key/sha256': tf.io.FixedLenFeature((), tf.string, default_value=''),

        f'rationale/{config["draw"]}/image/encoded': tf.io.FixedLenFeature((), tf.string, default_value=''),
        f'rationale/{config["draw"]}/image/format': tf.io.FixedLenFeature((), tf.string, default_value='jpeg'),
        f'rationale/{config["draw"]}/image/key/sha256': tf.io.FixedLenFeature((), tf.string, default_value=''),
    }

    keys_to_features.update({
        'label/answer_label': tf.io.FixedLenFeature((), tf.int64, -1),
        'answer/ctx': tf.io.VarLenFeature(tf.int64),
        'answer/ctx_tags': tf.io.VarLenFeature(tf.int64),

        'label/rationale_label': tf.io.FixedLenFeature((), tf.int64, -1),
        'rationale/ctx': tf.io.VarLenFeature(tf.int64),
        'rationale/ctx_tags': tf.io.VarLenFeature(tf.int64),
    })


    for i in range(4):
        keys_to_features.update({
            f'answer/choice_{i}': tf.io.VarLenFeature(tf.int64),
            f'answer/choice_tags_{i}': tf.io.VarLenFeature(tf.int64),

            f'rationale/choice_{i}': tf.io.VarLenFeature(tf.int64),
            f'rationale/choice_tags_{i}': tf.io.VarLenFeature(tf.int64),
        })

    return keys_to_features


def _process_items_to_handlers(config):
    items_to_handlers = {
        'img_id': (slim_example_decoder.Tensor('img_id')),
        'annot_id': (slim_example_decoder.Tensor('annot_id')),

        'answer/image':
            slim_example_decoder.Image(
                image_key=f'answer/{config["draw"]}/image/encoded',
                format_key=f'answer/{config["draw"]}/image/format',
                channels=3),
        'answer/key': (slim_example_decoder.Tensor(f'answer/{config["draw"]}/image/key/sha256')),

        'rationale/image':
            slim_example_decoder.Image(
                image_key=f'rationale/{config["draw"]}/image/encoded',
                format_key=f'rationale/{config["draw"]}/image/format',
                channels=3),
        'rationale/key': (slim_example_decoder.Tensor(f'rationale/{config["draw"]}/image/key/sha256')),
    }

    items_to_handlers.update({
        'answer_label': (slim_example_decoder.Tensor('label/answer_label')),
        'answer_ctx': (slim_example_decoder.Tensor('answer/ctx')),
        'answer_ctx_tags': (slim_example_decoder.Tensor('answer/ctx_tags')),

        'rationale_label': (slim_example_decoder.Tensor('label/rationale_label')),
        'rationale_ctx': (slim_example_decoder.Tensor('rationale/ctx')),
        'rationale_ctx_tags': (slim_example_decoder.Tensor('rationale/ctx_tags')),
    })

    for i in range(4):
        items_to_handlers.update({
            f'answer_choice_{i}': (slim_example_decoder.Tensor(f'answer/choice_{i}')),
            f'answer_choice_tags_{i}': (slim_example_decoder.Tensor(f'answer/choice_tags_{i}')),

            f'rationale_choice_{i}': (slim_example_decoder.Tensor(f'rationale/choice_{i}')),
            f'rationale_choice_tags_{i}': (slim_example_decoder.Tensor(f'rationale/choice_tags_{i}')),
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
    tensor_dict['answer/image'].set_shape([None, None, 3])
    tensor_dict['rationale/image'].set_shape([None, None, 3])
    return tensor_dict


def _dataset_parser(value, config, is_training):
    """Parse data to a fixed dimension input image and learning targets.
    """
    MAX_NUM_TOKENS = 184

    with tf.name_scope('parser'):
        data = _decode_record(value, config)

        # We'll build this one up
        features = {
            'img_id': encode_string(data['img_id'], 24),
            'annot_id': encode_string(data['annot_id'], 24),
        }

        # Normalize img
        image = tf.image.convert_image_dtype(data['answer/image'], dtype=tf.float32)
        data['answer/image'], data['answer/image_info'], _ = \
            resize_and_pad(image, config['image_size'], do_random_scale=is_training,
                           random_scale_max=1.1, random_scale_min=0.9)

        image = tf.image.convert_image_dtype(data['rationale/image'], dtype=tf.float32)
        data['rationale/image'], data['rationale/image_info'], _ = \
            resize_and_pad(image, config['image_size'], do_random_scale=is_training,
                           random_scale_max=1.1, random_scale_min=0.9)

        # bfloat16 and commit the image
        if config.get('use_bfloat16', False):
            tf.logging.info("image -> bfloat")
            data['answer/image'] = tf.cast(data['answer/image'], dtype=tf.bfloat16)
            data['rationale/image'] = tf.cast(data['rationale/image'], dtype=tf.bfloat16)

        features['images'] = tf.stack([data['answer/image'], data['rationale/image']])
        features['image_info'] = tf.stack([data['answer/image_info'], data['rationale/image_info']])

        # Handling text
        for i in range(4):
            data[f'answer_ctx_choice_{i}'] = tf.concat([tf.constant(VCR_prompt['answer'], dtype=tf.int32),
                                                        tf.cast(data['answer_ctx'], tf.int32),
                                                        tf.cast(data[f'answer_choice_{i}'], tf.int32),
                                                        tf.fill([1], END)], axis=0)

            data[f'rationale_ctx_choice_{i}'] = tf.concat([tf.constant(VCR_prompt['rationale'], dtype=tf.int32),
                                                           tf.cast(data['rationale_ctx'], tf.int32),
                                                           tf.cast(data[f'rationale_choice_{i}'], tf.int32),
                                                           tf.fill([1], END)], axis=0)

        ans_input = tf.stack([pad_to_fixed_size(data[f'answer_ctx_choice_{i}'],
                                                pad_value=PADDING, output_shape=[MAX_NUM_TOKENS]) for i in range(4)])
        rat_input = tf.stack([pad_to_fixed_size(data[f'rationale_ctx_choice_{i}'],
                                                pad_value=PADDING, output_shape=[MAX_NUM_TOKENS]) for i in range(4)])
        lm_input = tf.stack([ans_input, rat_input])
        lm_mask = tf.cast(tf.not_equal(lm_input, PADDING), tf.int32)

        features['lm_input'], features['lm_mask'] = lm_input, lm_mask

        ###########
        labels = {
            'lm_targets': tf.stack([data['answer_label'], data['rationale_label']]),
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
            for k in ['images', 'image_info', 'lm_input', 'lm_mask', 'lm_targets']:
                features[k] = tf.reshape(features[k], [batch_size * 2] + get_shape_list(features[k])[2:])

            for k in ['lm_targets']:
                labels[k] = tf.reshape(labels[k], [batch_size * 2] + get_shape_list(labels[k])[2:])

            if is_training and config.model['transpose_input']:
                # TPUs don't handle shapes with small numbers like "3" very well so we do this thing
                features['images'] = tf.transpose(features['images'], [1, 2, 3, 0])

            if is_training:
                # Turn [batch_size, 4, ...] things into [batch_size * 4]
                with tf.name_scope('flatten_examples'):
                    for k in ['lm_input', 'lm_mask']:
                        features[k] = tf.reshape(features[k], [batch_size * 2 * 4] + get_shape_list(features[k])[2:])
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

    use_train = True
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
