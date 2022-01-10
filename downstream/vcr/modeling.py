import os
import math

import tensorflow as tf

from model.modeling import MerlotModel
from utils import optimization
from utils.model_utils import get_assignment_map_from_checkpoint, get_shape_list, construct_host_call, gelu, dropout
from utils.neat_config import NeatConfig
from utils.transformer import create_initializer


def cls_metric(logits_flat, targets):
    cls_log_probs = tf.reshape(tf.nn.log_softmax(logits_flat, axis=-1), [tf.shape(targets)[0], -1])
    predictions = tf.argmax(cls_log_probs, axis=-1)
    accuracy = tf.cast(tf.equal(targets, predictions), tf.float32)
    return {'predictions': predictions,
            'log_prob': cls_log_probs,
            'labels': targets,
            'accuracy': accuracy}


def model_fn_builder(config: NeatConfig):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        if is_training and config.model.get('transpose_input', False):
            tf.logging.info("Transpose images")
            features['images'] = tf.transpose(features['images'], [3, 0, 1, 2])

        if not is_training:
            batch_size, _, lang_seq_length = get_shape_list(features['lm_input'], expected_rank=3)
            # Turn [batch_size, 4, ...] things into [2*batch_size]
            with tf.name_scope('flatten_examples'):
                for k in ['lm_input', 'lm_mask']:
                    features[k] = tf.reshape(features[k], [batch_size * 4] + get_shape_list(features[k])[2:])

        num_texts = config.model["num_texts"]

        model = MerlotModel(
            config=config.model,
            is_training=is_training,
            image=features['images'],
            input_ids=features['lm_input'],
            use_tpu=config.device['use_tpu'],
            mask_input=False,
        )

        def cls_head_val(hidden_state, bias_pi=0.25):
            first_token_tensor = hidden_state[:, 0, :]
            with tf.variable_scope(f'{config.downstream["mode"]}_cls'):
                logits = tf.layers.dense(
                    first_token_tensor,
                    config.model['hidden_size'] // 2,
                    activation=gelu,
                    kernel_initializer=create_initializer(config.model['initializer_range']),
                    name='classifier_mlp0',
                )
                logits = tf.layers.dense(
                    logits,
                    1,
                    kernel_initializer=create_initializer(config.model['initializer_range']),
                    bias_initializer=tf.constant_initializer(-math.log((1 - bias_pi) / bias_pi)),
                    name='classifier_mlp1',
                )
            logits_flat = tf.reshape(logits, [model.img_batch_size, 4])
            return logits_flat

        def cls_head(hidden_state, bias_pi=0.25):
            first_token_tensor = hidden_state[:, 0, :]
            first_token_tensor = tf.reshape(first_token_tensor, [model.img_batch_size , num_texts, -1])
            first_token_tensor = tf.reshape(first_token_tensor, [model.img_batch_size // 2, 2, num_texts, -1])

            ans_token_tensor, rat_token_tensor = first_token_tensor[:, 0], first_token_tensor[:, 1]
            ans_token_tensor = tf.reshape(ans_token_tensor, [-1, config.model['hidden_size']])
            rat_token_tensor = tf.reshape(rat_token_tensor, [-1, config.model['hidden_size']])

            with tf.variable_scope('answer_cls'):
                ans_token_tensor = dropout(ans_token_tensor, dropout_prob=config.model['hidden_dropout_prob'])
                ans_logits = tf.layers.dense(
                    ans_token_tensor,
                    config.model['hidden_size'] // 2,
                    activation=gelu,
                    kernel_initializer=create_initializer(config.model['initializer_range']),
                    name='classifier_mlp0',
                )
                ans_logits = dropout(ans_logits, dropout_prob=config.model['hidden_dropout_prob'])
                ans_logits = tf.layers.dense(
                    ans_logits,
                    1,
                    kernel_initializer=create_initializer(config.model['initializer_range']),
                    bias_initializer=tf.constant_initializer(-math.log((1 - bias_pi) / bias_pi)),
                    name='classifier_mlp1',
                )
                ans_logits_flat = tf.reshape(ans_logits, [model.img_batch_size // 2, 4])

            with tf.variable_scope('rationale_cls'):
                rat_token_tensor = dropout(rat_token_tensor, dropout_prob=config.model['hidden_dropout_prob'])
                rat_logits = tf.layers.dense(
                    rat_token_tensor,
                    config.model['hidden_size'] // 2,
                    activation=gelu,
                    kernel_initializer=create_initializer(config.model['initializer_range']),
                    name='classifier_mlp0',
                )
                rat_logits = dropout(rat_logits, dropout_prob=config.model['hidden_dropout_prob'])
                rat_logits = tf.layers.dense(
                    rat_logits,
                    1,
                    kernel_initializer=create_initializer(config.model['initializer_range']),
                    bias_initializer=tf.constant_initializer(-math.log((1 - bias_pi) / bias_pi)),
                    name='classifier_mlp1',
                )
                rat_logits_flat = tf.reshape(rat_logits, [model.img_batch_size // 2, 4])

            logits_flat = tf.concat([ans_logits_flat, rat_logits_flat], axis=1)
            logits_flat = tf.reshape(logits_flat, [model.img_batch_size, 4])

            return logits_flat

        output_head = cls_head if is_training else cls_head_val
        hidden_state = tf.cast(model.encoder_hidden_states['lang'], dtype=tf.float32)
        model.logits_flat = output_head(hidden_state)

        def cls_loss(target):
            """
            Get the loss for LM.
            :param targets: [batch_size, num_answers] float32 the TARGETS.
            :return: loss.
            """
            one_hot_labels = tf.one_hot(target, depth=4, dtype=model.logits_flat.dtype)
            per_example_loss = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_labels, logits=model.logits_flat)

            loss = tf.reduce_sum(per_example_loss) / tf.cast(model.img_batch_size, tf.float32)
            return {'loss': loss}

        # Loss
        losses = cls_loss(labels['lm_targets'])
        eval_fn, metric = cls_metric, 'accuracy'
        evaluation = eval_fn(model.logits_flat, labels['lm_targets'])

        if is_training:
            tvars = [x for x in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if 'global_step' not in x.name]
        else:
            tvars = tf.trainable_variables()

        ckpt_to_assignment_map = {}
        initialized_variable_names = {}

        init_checkpoint = config.model.get('init_checkpoint', None)
        if init_checkpoint:
            regular_assignment_map, regular_initialized_variable_names = get_assignment_map_from_checkpoint(
                tvars, init_checkpoint=init_checkpoint
            )
            ckpt_to_assignment_map['regular'] = regular_assignment_map
            initialized_variable_names.update(regular_initialized_variable_names)

        def scaffold_fn():
            """Loads pretrained model through scaffold function."""
            # ORDER BY PRIORITY
            for ckpt_type, ckpt in [('regular', init_checkpoint)]:
                if ckpt:
                    tf.train.init_from_checkpoint(ckpt, ckpt_to_assignment_map[ckpt_type])
            return tf.train.Scaffold()

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

        train_op, train_metrics = optimization.build_optimizer_from_config(
            loss=losses['loss'],
            optimizer_config=config.optimizer,
            device_config=config.device,
        )
        train_metrics[metric] = tf.reduce_mean(evaluation[metric])
        train_metrics.update(losses)

        host_call = construct_host_call(scalars_to_log=train_metrics,
                                        model_dir=config.device['output_dir'] if mode != tf.estimator.ModeKeys.EVAL
                                        else os.path.join(config.device['output_dir'], 'eval'),
                                        iterations_per_loop=config.device.get('iterations_per_loop', 1000))

        # This could be useful for debugging, but we can take it out.
        if mode == tf.estimator.ModeKeys.PREDICT:
            evaluation.update({'annot_id': features['annot_id']})
            return tf.contrib.tpu.TPUEstimatorSpec(mode=tf.estimator.ModeKeys.PREDICT,
                                                   predictions=evaluation)
        elif mode == tf.estimator.ModeKeys.EVAL:
            def metric_fn(measure):
                avg_acc = tf.metrics.mean(measure, name='avg')
                return {'avg': avg_acc}
            return tf.contrib.tpu.TPUEstimatorSpec(mode=tf.estimator.ModeKeys.EVAL,
                                                   loss=losses['loss'],
                                                   train_op=train_op,
                                                   eval_metrics=(metric_fn, [evaluation[metric]]),
                                                   scaffold_fn=scaffold_fn,
                                                   host_call=host_call,
                                                   predictions=evaluation)

        return tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode,
            loss=losses['loss'],
            train_op=train_op,
            eval_metrics=None,
            scaffold_fn=scaffold_fn,
            host_call=host_call)

    return model_fn