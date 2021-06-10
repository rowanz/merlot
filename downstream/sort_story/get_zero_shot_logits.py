"""Prediction generator"""
import sys
sys.path.append('../../')
import numpy as np
import tqdm
from utils.model_utils import get_shape_list
import os
import h5py
import tensorflow as tf

from downstream.sort_story.dataloader import input_fn_builder
from model.modeling import MerlotModel

from utils.neat_config import NeatConfig
SPLIT_NAME = 'val'
config = NeatConfig.from_args("Get ZS logits for unsupervised sortstory", default_config_file='../../model/configs/merlot_5segments.yaml')

# You need to fill these in in that file
assert tf.gfile.Exists(os.path.join(config.device['output_dir'], 'checkpoint'))
assert tf.gfile.Exists(os.path.join(config.device['output_dir'], 'model.ckpt.meta'))
assert tf.gfile.Exists(os.path.join(config.device['output_dir'], 'model.ckpt.index'))
assert tf.gfile.Exists(os.path.join(config.device['output_dir'], 'model.ckpt.data-00000-of-00001'))

config.data['val_file'] = os.environ['SORTSTORY_PATH']

NUM_CHUNKS = 5
config.data['num_chunks'] = NUM_CHUNKS
assert config.model['num_chunks_in_group'] == NUM_CHUNKS


# This gives you multiple samples of the permutation order
# We'll use 2 and average the results just in case it is sensitive to the exact "unk1/unk2/etc." tokens that we use
# Using > 2 actually doesn't seem to help
duplication_factor = 2

def model_fn(features, labels, mode, params):
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
        tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    assert not is_training
    images_to_use = tf.tile(features['images'], [duplication_factor, 1, 1, 1, 1])
    sents_to_use = tf.tile(features['sentences'], [duplication_factor, 1, 1])

    with tf.name_scope('flatten_examples'):
        # Flatten bsize and transpose
        batch_size_, num_imgs_per_batch, h, w, three_ = get_shape_list(images_to_use, 5)
        images_resh = tf.reshape(images_to_use, [batch_size_ * num_imgs_per_batch, h, w, 3])
        input_ids = sents_to_use[:, :, :32]

    # Fix the random seed
    shuffled_idx_img = tf.cast(tf.argsort(tf.reshape(
        tf.random.stateless_uniform([batch_size_ * NUM_CHUNKS], seed=[123,1234]), [batch_size_, NUM_CHUNKS]), 1), dtype=tf.int32) + 64

    model = MerlotModel(
        config=config.model,
        is_training=False,
        image=images_resh,
        input_ids=input_ids,
        use_tpu=config.device['use_tpu'],
        mask_input=False,
        shuffled_idx_img=shuffled_idx_img,
    )

    h_lang = tf.reshape(model.encoder_hidden_states['lang'],
                        [model.B, model.num_chunks_in_group, model.lang_chunk_length, model.hidden_size])[:, :, 0]
    h_viz = tf.reshape(model.encoder_hidden_states['viz'],
                       [model.B, model.num_chunks_in_group, model.viz_chunk_length, model.hidden_size])[:, :, 0]

    # Score both language-vision and vision-vision unshuffling, even though we never use vision-vision
    modality_pairs = [
        {'name': 'lang_viz', 'xa': h_lang, 'xb': h_viz},
        {'name': 'viz_viz', 'xa': h_viz, 'xb': h_viz},
    ]
    for x in modality_pairs:
        allpairs_logits = model.allpairs_temporal_logits(xa=x['xa'], xb=x['xb'], scope_name='{}_temporal'.format(x['name']))
        probs = tf.nn.softmax(allpairs_logits, -1)[:, 1:]

        # Average over different versions of the mask
        probs = tf.reshape(probs, [params['batch_size'], duplication_factor, config.model['num_chunks_in_group'], config.model['num_chunks_in_group'], 3])
        probs = tf.reduce_mean(probs, 1)

        features['{}_probs'.format(x['name'])] = probs

    features['images'] = tf.cast(features['images'], dtype=tf.float32)
    del features['images']
    return tf.contrib.tpu.TPUEstimatorSpec(mode=tf.estimator.ModeKeys.PREDICT, predictions=features)


estimator = tf.contrib.tpu.TPUEstimator(
    use_tpu=config.device['use_tpu'],
    model_fn=model_fn,
    config=config.device['tpu_run_config'],
    train_batch_size=config.device['train_batch_size'],
    eval_batch_size=config.device['val_batch_size'],
    predict_batch_size=config.device['val_batch_size'],
)

with h5py.File(f'logits_{SPLIT_NAME}.h5', 'w') as h5:
    for x in tqdm.tqdm(estimator.predict(input_fn=input_fn_builder(config, is_training=False), yield_single_examples=True)):
        try:
            grp = h5.create_group(str(x['story_id']))
        except ValueError as e:
            # group exists
            print(str(e))
            continue
        grp.create_dataset('permutation_identity_encode', data=x['permutation_identity_encode'])
        if 'images' in x:
            grp.create_dataset('images', data=(255 * x['images']).astype(np.uint8))
        grp.create_dataset('sentences', data=x['sentences'])
        for modality_name in ['lang_viz', 'viz_viz']:
            grp.create_dataset(f'{modality_name}_probs', data=x[f'{modality_name}_probs'])