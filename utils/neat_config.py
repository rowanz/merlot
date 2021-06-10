"""
Just trying this out. essentially this utility is going to force you to pass in a config file whenever you do anything

including debugging

Essentially there are 4 main components: data, model, optimizer, and device. (For other things we have a 'misc') option.
"""
import argparse
import inspect
import os
from collections import defaultdict
from copy import deepcopy
from datetime import datetime

import pytz
import tensorflow as tf
import yaml


class NeatConfig(object):
    def __init__(self):
        self.data = {}
        self.model = {}
        self.optimizer = {}
        self.device = {}
        self.downstream = {}
        self.validate = {}
        self.misc = {}

    @classmethod
    def from_yaml(cls, config_file):
        """
        Sets up config from a yaml file
        :param config_file: where 2 load from
        :return:
        """
        with tf.gfile.Open(config_file, 'r') as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)

        print("~~~~~\nLOADED CONFIG FROM {}\n~~~~~\n".format(config_file), flush=True)
        return cls.from_dict(config_dict, orig_config_file=config_file)

    @classmethod
    def from_dict(cls, config_dict, orig_config_file=None):
        """
        Loads from dict
        :param config_dict:
        :param orig_config_file: Where it was originally loaded from, if we want to print a helpful message
        :return:
        """
        config = deepcopy(config_dict)
        if 'misc' not in config:
            config['misc'] = {}
        # Mandatory keys
        for key in ['data', 'model', 'optimizer', 'device']:
            if key not in config:
                raise ValueError("Configuration file {} is missing {}".format(orig_config_file, key))

        # Save config to the cloud, if the output directory is there
        if 'output_dir' not in config['device']:
            raise ValueError("Missing output directory")
        print("~~~\nWILL WRITE TO {}\n~~~\n".format(config['device']['output_dir']), flush=True)

        # Special handling of TPUs. I think this should work on GPUs too but haven't checked
        # if config['device']['use_tpu']:
        config['device']['tpu_run_config'] = get_tpu_run_config(config['device'])

        if config['misc'].get('verbose', True):
            tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

        # Special handling of globs in the data files
        for x in ['train_file', 'val_file', 'test_file']:
            if x in config['data']:
                v_orig = config['data'][x]

                v_list = []
                matches = defaultdict(list)
                for input_pattern in config['data'][x].split(','):
                    for fn in tf.io.gfile.glob(input_pattern):
                        v_list.append(fn)
                        matches[input_pattern].append(fn)

                if sum(len(v) for v in matches.values()) == len(v_list):
                    # Get a friendly printout
                    input_file_tree = []
                    for prefix, match_list in matches.items():
                        input_file_tree.append('{}[{}]'.format(prefix, ','.join(match_list)))
                    input_file_tree = '\n'.join(input_file_tree)

                    # These can get kinda long and unwieldy
                    if len(input_file_tree) > 100:
                        input_file_tree = input_file_tree[:100] + '...({} total)'.format(len(v_list))

                    print("Input files: {} ->\n{}\n\n".format(v_orig, input_file_tree), flush=True)
                else:
                    print("Input files: {} ->\n{}\n\n".format(v_orig, '   '.join(v_list)), flush=True)
                config['data'][f'{x}_expanded'] = v_list

        config_cls = cls()
        config_cls.__dict__.update(config)
        return config_cls

    @classmethod
    def from_args(cls, help_message="NeatConfig", default_config_file=None):
        parser = argparse.ArgumentParser(description=help_message)
        parser.add_argument(
            'config_file',
            nargs='?',
            help='Where the config.yaml is located',
            default=default_config_file,
            type=str,
        )
        args = parser.parse_args()
        if not args.config_file:
            raise ValueError("No config file provided!")

        if not tf.io.gfile.exists(args.config_file):
            raise ValueError("Config file {} not found?".format(args.config_file))
        return cls.from_yaml(args.config_file)


def get_tpu_run_config(device_config):
    """
    Sets up the tpu
    :param device_config: The part of the config file that is
    :return:
    """
    tpu_cluster_resolver = None
    tpu_name = device_config.get('tpu_name', os.uname()[1])  # This is the hostname

    if device_config['use_tpu']:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            tpu_name, zone=device_config.get('tpu_zone', None), project=device_config.get('gcp_project', None))
        tf.compat.v1.Session.reset(tpu_cluster_resolver.get_master())

    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=device_config.get('master', None),
        model_dir=device_config['output_dir'],
        save_checkpoints_steps=device_config.get('iterations_per_loop', 1000),
        keep_checkpoint_max=None,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=device_config.get('iterations_per_loop', 1000),
            # num_shards=device_config.get('num_tpu_cores', 8), # Commented out because it's always 8 for testing.
            per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2,
            #experimental_host_call_every_n_steps=device_config.get('experimental_host_call_every_n_steps', 1),
        ))

    if tf.io.gfile.exists(device_config['output_dir']):
        print(f"The output directory {device_config['output_dir']} exists!")
    return run_config
