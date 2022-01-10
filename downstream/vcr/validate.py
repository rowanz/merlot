""" Validating script! """
import os
import sys
sys.path.append('../../')

import tensorflow as tf
from downstream.vcr import dataloader
from downstream.vcr.modeling import model_fn_builder
from utils.neat_config import NeatConfig

config = NeatConfig.from_args("Pretraining script", default_config_file='../../model/configs/merlot_vcr.yaml')
model_fn = model_fn_builder(config)

estimator = tf.contrib.tpu.TPUEstimator(
    use_tpu=config.device['use_tpu'],
    model_fn=model_fn,
    config=config.device['tpu_run_config'],
    train_batch_size=config.device['train_batch_size'],
    eval_batch_size=config.device['val_batch_size'],
    predict_batch_size=config.device['val_batch_size'],
)

def terminate_eval():
    tf.logging.info('Terminating eval after %d seconds of no checkpoints' % config.validate['eval_timeout'])
    return True

# Run evaluation when there's a new checkpoint
for ckpt in tf.contrib.training.checkpoints_iterator(
        checkpoint_dir=config.device['output_dir'],
        timeout=config.validate['eval_timeout'],
        timeout_fn=terminate_eval):

    tf.logging.info('Starting to evaluate.')
    try:
        # Note that if the eval_samples size is not fully divided by the
        # eval_batch_size. The remainder will be dropped and result in
        # differet evaluation performance than validating on the full set.
        results = estimator.evaluate(input_fn=dataloader.input_fn_builder(config, is_training=False),
                                     steps=config.validate['eval_samples'] // config.device['val_batch_size'])
        tf.logging.info('Eval results: %s' % results)

        writer = tf.summary.FileWriter(os.path.join(config.device['output_dir'], 'eval'),
                                       tf.get_default_graph())
        summary = tf.Summary(value=[tf.Summary.Value(tag='avg_{}'.format('accuracy'), simple_value=results['avg'])])
        writer.add_summary(summary, results['global_step'])

        # Terminate eval job when final checkpoint is reached
        current_step = int(os.path.basename(ckpt).split('-')[1])
        total_step = config.optimizer['num_train_steps']
        if current_step >= total_step:
            tf.logging.info('Evaluation finished after training step %d' % current_step)
            break

    except tf.errors.NotFoundError:
        # Since the coordinator is on a different job than the TPU worker,
        # sometimes the TPU worker does not finish initializing until long after
        # the CPU job tells it to start evaluating. In this case, the checkpoint
        # file could have been deleted already.
        tf.logging.info('Checkpoint %s no longer exists, skipping checkpoint' % ckpt)
