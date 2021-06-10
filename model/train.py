""" Training script! """
import sys
sys.path.append('../')

import tensorflow as tf

from utils.neat_config import NeatConfig

config = NeatConfig.from_args("Pretraining script", default_config_file='configs/v0a-base.yaml')

from model import dataloader

from model.modeling import model_fn_builder

model_fn = model_fn_builder(config)

estimator = tf.contrib.tpu.TPUEstimator(
    use_tpu=config.device['use_tpu'],
    model_fn=model_fn,
    config=config.device['tpu_run_config'],
    train_batch_size=config.device['train_batch_size'],
    eval_batch_size=config.device['val_batch_size'],
    predict_batch_size=config.device['val_batch_size'],
)
estimator.train(input_fn=dataloader.input_fn_builder(config, is_training=True),
                max_steps=config.optimizer['num_train_steps'])
