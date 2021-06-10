import re
from collections import defaultdict
from copy import deepcopy

import numpy as np
import tensorflow as tf

from utils.model_utils import get_shape_list


def build_optimizer_from_config(loss, optimizer_config, device_config=None):
    """
    This is a utility to build an optimizer from optimizer_config.

    :param loss: what to use.
    :param optimizer_config: k/v of options
    :param device_config: Additional options that can be rolled in
    :return: An optimizer
    """
    optimizer_types = {
        'adam_optimizer': create_fixed_adam_optimizer_with_warmup,
    }
    if optimizer_config['type'] not in optimizer_types:
        raise ValueError("The optimizer type {} isn't supported".format(optimizer_config['type']))

    kwargs = deepcopy(optimizer_config)
    if device_config is not None:
        kwargs.update(deepcopy(device_config))
    del kwargs['type']
    return optimizer_types[optimizer_config['type']](loss, **kwargs)


def _print_var_list_for_debugging(var_list):
    """
    For debugging, print a list of vars. Sort by the shapes, also print the total size.
    :param var_list: list of vars.
    :return: Nothing!
    """
    if len(var_list) == 0:
        tf.logging.info('~~~ (N/A) ~~~')
        return
    sorted_vars = sorted([(_get_variable_name(x.name), tuple(get_shape_list(x))) for x in var_list],
                         key=lambda x: -np.prod(x[1]))
    total_size = sum([np.prod(x[1]) for x in sorted_vars])
    # Pretty print each line
    longest_name = max([len(x[0]) for x in sorted_vars])
    prints = [' {s:<{w}}'.format(s=x[0], w=longest_name) + '{}'.format(x[1]) for x in sorted_vars]
    for l in prints:
        tf.logging.info(l)
    tf.logging.info('~~~~ Total size = {} or {:.1f}M\n'.format(
        total_size, float(total_size) / 1000000.0
    ))


def create_fixed_adam_optimizer_with_warmup(loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu,
                                            weight_decay_rate=1e-4, param_overrides=None, freeze_scope=None,
                                            verbose=False, clip_norm=1.0, adafactor=False, epsilon=1e-6, beta_2=0.98,
                                            use_bfloat16_adam=False, do_param_scale=False, decay_beta2_adafactor=False,
                                            **kwargs):
    """
    Does AdamW optimization. Unlike the BERT optimizer, here I added bias correct which the original
    one didn't seem to have.

    :param loss:
    :param learning_rate: The default learning rate we'll use. All of the learning rates, including overridden ones
                          will get scaled during the initial `num_warmup_steps`.
    :param num_train_steps: How many steps to train for overall.
    :param num_warmup_steps: A number, presumably < num_train_steps which specifies for how long we warmup.
    :param use_tpu: Whether to use TPU. This is important because we need to duplicate the optimizer accross shards.
    :param weight_decay_rate: How much to decay the weights by default.
    :param param_overrides: Which parameters to override. This works like the following. You pass in a
                            LIST of LIST, DICTIONARY pairs. Each pair consists of a bunch of regular expressions
                            and if one of those are activated, we will override the default parameters in that instance.
                            For instance

                            ["LayerNorm", "layer_norm", 'GroupNorm', "bias"], {"weight_decay_rate": 0}

                            will set any parameter matching the first couple of regexes to have weight_decay_rate of 0.
    :param freeze_scope: OLD deprecated parameter that sets anything matching ["^freeze_scope/"] to have {"learning_rate": 0}
    :param verbose: Use this for extra debugging output
    :param kwargs: extra args, not needed
    :return:
    """

    global_step = tf.compat.v1.train.get_or_create_global_step()

    # Implements linear decay of the learning rate. This does it globally over all parameters
    # which should be OK.

    # Make it so that we scale the loss UP to learning_rate
    # scale * (1-(num_warmup_steps / num_train_steps)) = 1.0
    # scale = 1/(1-(num_warmup_steps / num_train_steps))
    # scale = num_train_steps /(num_train_steps - num_warmup_steps
    base_scale = float(num_train_steps) / (
                float(num_train_steps) - float(num_warmup_steps) + 1.0) if num_warmup_steps else 1.0
    learning_rate_scale = tf.compat.v1.train.polynomial_decay(
        tf.constant(value=base_scale, shape=[], dtype=tf.float32),
        global_step,
        num_train_steps,
        end_learning_rate=0.0,
        power=1.0,
        cycle=False)

    # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
    # learning rate will be `global_step/num_warmup_steps * learning_rate`.
    if num_warmup_steps:
        global_steps_int = tf.cast(global_step, tf.int32)
        warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

        global_steps_float = tf.cast(global_steps_int, tf.float32)
        warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

        warmup_percent_done = global_steps_float / warmup_steps_float

        learning_rate_scale = tf.where(global_steps_int < warmup_steps_int, warmup_percent_done, learning_rate_scale)

    # Deal with the parameter overrides.
    # We can override:
    #     learning_rate. if learning_rate = 0 then we aren't training it at all.
    #     beta_1
    #     beta_2
    #     epsilon
    #     weight_decay_rate

    if param_overrides is None:
        param_overrides = []

    if freeze_scope is not None:
        print("NOTE! freeze_scope is deprecated. You can do the exact same thing by instead setting\n"
              "param_overrides: [[[\"^{}\"], {{\"learning_rate\": 0}}]]".format(freeze_scope))
        param_overrides.append([[f'^{freeze_scope}'], {'learning_rate': 0}])

    tvars = tf.trainable_variables()
    param_name_to_overridden_parameters = defaultdict(dict)
    for regexes, overridden_parameters in param_overrides:
        for k in overridden_parameters:
            if k not in ('learning_rate', 'weight_decay_rate', 'beta_1', 'beta_2', 'epsilon', 'do_factor'):
                raise ValueError(
                    "Regex rule {} -> {} isn't OK because {} isn't a changable optimization parameter".format(
                        regexes, overridden_parameters, k
                    ))

        for regex in regexes:
            for p in tvars:
                param_name = _get_variable_name(p.name)
                if re.search(regex, param_name) is not None:
                    param_name_to_overridden_parameters[param_name].update(overridden_parameters)

    non_trainable_vars = [v for v in tvars
                          if not param_name_to_overridden_parameters[_get_variable_name(v.name)].get('learning_rate',
                                                                                                     1.0)]
    if len(non_trainable_vars) != 0:
        tf.logging.info("\n~~~~~ NOT training the following variables:")
        _print_var_list_for_debugging(non_trainable_vars)
        tvars = [v for v in tvars
                 if param_name_to_overridden_parameters[_get_variable_name(v.name)].get('learning_rate', 1.0)]

    # Get all possible conditions, just for debugging purposes.
    conditions_to_params = defaultdict(list)
    for v in tvars:
        conditions = param_name_to_overridden_parameters[_get_variable_name(v.name)]
        conditions_str = ','.join(f'{k}={v}' for k, v in sorted(conditions.items()))
        conditions_to_params[conditions_str].append(v)

    for conditions, param_list in conditions_to_params.items():
        if not conditions:
            tf.logging.info(
                "\n~~~~~ For the following params, using DEFAULTS \n{}".format(','.join(f'{k}={v}' for k, v in {
                    'learning_rate': learning_rate, 'weight_decay_rate': weight_decay_rate, 'beta_1': 0.9,
                    'beta_2': beta_2, 'eps': epsilon, 'use_bfloat16_adam': use_bfloat16_adam,
                }.items())))
        else:
            tf.logging.info("\nFor the following params, overriding {}".format(conditions))
        _print_var_list_for_debugging(param_list)

    grads = tf.gradients(loss, tvars)

    if adafactor:
        raise ValueError("Adafactor not supported rn")
    else:
        optimizer = AdamOptimizer(
            learning_rate=learning_rate,
            weight_decay_rate=weight_decay_rate,
            learning_rate_scale=learning_rate_scale,
            beta_1=0.9,
            beta_2=beta_2,
            epsilon=epsilon,
            param_name_to_overridden_parameters=dict(param_name_to_overridden_parameters),
            make_things_dependent_on_grad=True,
            use_bfloat16_adam=use_bfloat16_adam,
        )

    train_metrics = {
        'learning_rate': learning_rate * learning_rate_scale,
        'minibatch_loss': loss,
    }

    if verbose:
        for v in tvars:
            if v.dtype == tf.bfloat16:
                raise ValueError(f"{v.name} is bfloat16")
        train_metrics['weight_decay_loss'] = tf.add_n([
            tf.nn.l2_loss(v) * param_name_to_overridden_parameters[
                _get_variable_name(v.name)].get('weight_decay_rate', weight_decay_rate)
            for v in tvars])

        # Clip grads AND log
        param_to_l2 = {_get_variable_name(x.name): tf.nn.l2_loss(y) for x, y in zip(tvars, grads) if y is not None}
        global_norm = tf.math.sqrt(2.0 * tf.add_n(list(param_to_l2.values())))

        if clip_norm > 0.0:
            tf.logging.info("clipping the global norm to {:.3f}".format(clip_norm))
            (grads, _) = tf.clip_by_global_norm(grads, use_norm=global_norm, clip_norm=clip_norm)
        else:
            tf.logging.info("Not clipping the global norm")

        # Log the global norms. I'm not worrying about grouping or any of that
        # so for language/layer00/key_layer/kernel
        #    and language/layer00/key_layer/bias
        # we log both these parameters as well as language/layer00/key_layer/, language/layer00/ ...
        all_groups = sorted(set(['/'.join(x.split('/')[:(depth + 1)]) for x in param_to_l2.keys()
                                 for depth in range(len(x.split('/')))]))

        for g in all_groups:
            # Hide some boring things
            if g.split('/')[-1] in ('beta', 'kernel', 'bias', 'gamma'):
                continue

            train_metrics[f'gradnorms/{g}'] = tf.math.sqrt(
                2.0 * tf.add_n([v for k, v in param_to_l2.items() if k.startswith(g)]))
        train_metrics[f'gradnorms/_overall'] = global_norm
    else:
        # Clip by global norm. I think we need this, but RoBERTa didn't use it so maybe not? idk. adding it anyways
        if clip_norm > 0.0:
            tf.logging.info("clipping the global norm to {:.3f}".format(clip_norm))
            grads, use_norm = tf.clip_by_global_norm(grads, clip_norm=clip_norm)
            train_metrics[f'gradnorms/_overall'] = use_norm
        else:
            tf.logging.info("Not clipping the global norm")

    if use_tpu:
        optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

    train_op = optimizer.apply_gradients(
        zip(grads, tvars), global_step=global_step)

    # Normally the global step update is done inside of `apply_gradients`.
    # However, `AdamOptimizer` doesn't do this. But if you use
    # a different optimizer, you should probably take this line out.
    # + If you're using BN you need UPDATE_OPS to run also
    new_global_step = global_step + 1
    train_op = tf.group(train_op, [global_step.assign(new_global_step)],
                        tf.get_collection(tf.GraphKeys.UPDATE_OPS))

    return train_op, train_metrics


def _get_variable_name(param_name):
    """Get the variable name from the tensor name. This just strips off the trailing :0"""
    m = re.match("^(.*):\\d+$", param_name)
    if m is not None:
        param_name = m.group(1)
    return param_name


# extreme hacky stuff
missing_precision = 1.00390625 # 1 / (2 ** 8)
def _decode_v(stored_v):
    """
    Use the extra bit to get 1 extra point of range
    If we do this hack then we will be off by at most 1 / (2 ** 9) which is better I guess
    If sign bit is positive do nothing
    If sign bit is negative multiply
    :param stored_v:
    :param use_bfloat16:
    :return:
    """
    sign = tf.math.sign(stored_v)  # [1 or -1]
    v_abs = tf.cast(tf.abs(stored_v), dtype=tf.float32)
    v_abs = tf.where(tf.greater(sign, 0), v_abs, v_abs * missing_precision)
    return v_abs

def _encode_v(stored_v):
    bfloat_enc = tf.cast(stored_v, dtype=tf.bfloat16)
    bfloat_enc_f32 = tf.cast(bfloat_enc, dtype=tf.float32)
    err0 = tf.abs(bfloat_enc_f32 - stored_v)
    err1 = tf.abs(bfloat_enc_f32 * missing_precision - stored_v)
    return tf.where(tf.less_equal(err0, err1), bfloat_enc, -bfloat_enc)

class AdamOptimizer(tf.compat.v1.train.Optimizer):
    """A basic Adam optimizer that includes "correct" L2 weight decay.
    Also adding bias correction
    """

    def __init__(self,
                 learning_rate,
                 learning_rate_scale=1.0,
                 weight_decay_rate=0.0,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-6,
                 param_name_to_overridden_parameters=None,
                 name="AdamOptimizer",
                 make_things_dependent_on_grad=False,
                 use_bfloat16_adam=False,
                 do_param_scale=False,
                 decay_beta2_adafactor=False):
        """Constructs a AdamWeightDecayOptimizer."""
        super(AdamOptimizer, self).__init__(False, name)

        self.learning_rate = learning_rate
        self.learning_rate_scale = learning_rate_scale
        self.weight_decay_rate = weight_decay_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.param_name_to_overridden_parameters = {} if param_name_to_overridden_parameters is None else param_name_to_overridden_parameters
        self.make_things_dependent_on_grad = make_things_dependent_on_grad
        self.use_bfloat16_adam=use_bfloat16_adam
        self.do_param_scale = do_param_scale
        self.do_factor = True  # won't do anything unless adafactor is on
        self.decay_beta2_adafactor = decay_beta2_adafactor

    def _get_hyperparam(self, param_name, hyperparam_name):
        """
        For the given parameter, get the right hyperparameter. It might have been overridden.
        :param param_name:
        :param hyperparam_name:
        :return:
        """
        if hyperparam_name not in ('learning_rate', 'weight_decay_rate', 'beta_1', 'beta_2', 'epsilon', 'do_factor'):
            raise ValueError(f"Invalid hyperparameter name {hyperparam_name}")
        if param_name not in self.param_name_to_overridden_parameters:
            return getattr(self, hyperparam_name)
        overridden_params = self.param_name_to_overridden_parameters[param_name]

        return overridden_params.get(hyperparam_name, getattr(self, hyperparam_name))

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """See base class."""
        assignments = []
        for (grad, param) in grads_and_vars:
            if grad is None or param is None:
                continue

            param_name = _get_variable_name(param.name)
            # Override parameters
            beta_1 = self._get_hyperparam(param_name, 'beta_1')
            beta_2 = self._get_hyperparam(param_name, 'beta_2')
            weight_decay_rate = self._get_hyperparam(param_name, 'weight_decay_rate')
            epsilon = self._get_hyperparam(param_name, 'epsilon')
            learning_rate = self._get_hyperparam(param_name, 'learning_rate') * self.learning_rate_scale

            # Bias correction
            t = tf.cast(global_step, dtype=tf.float32) + 1.0
            bc1 = 1.0 - tf.pow(beta_1, t)
            bc2 = 1.0 - tf.pow(beta_2, t)
            learning_rate *= tf.sqrt(bc2) / bc1

            grad_squared = tf.square(grad) + 1e-30

            if self.make_things_dependent_on_grad:
                # HACK: Make things dependent on grad.
                # This confounds the XLA rewriter and keeps it from fusing computations
                # across different variables.  This fusion is a bad for HBM usage, since
                # it causes the gradients to persist in memory.
                grad_squared_mean = tf.reduce_mean(grad_squared)
                learning_rate += grad_squared_mean * 1e-30
                epsilon += grad_squared_mean * 1e-30

            dtype = tf.bfloat16 if self.use_bfloat16_adam else tf.float32
            stored_m = tf.get_variable(
                name=param_name + "/adam_m",
                shape=param.shape.as_list(),
                dtype=dtype,
                trainable=False,
                initializer=tf.zeros_initializer())
            stored_v = tf.get_variable(
                name=param_name + "/adam_v",
                shape=param.shape.as_list(),
                dtype=dtype,
                trainable=False,
                initializer=tf.zeros_initializer())

            m = tf.cast(stored_m, dtype=tf.float32) if self.use_bfloat16_adam else stored_m
            v = _decode_v(stored_v) if self.use_bfloat16_adam else stored_v

            # Standard Adam update.
            next_m = tf.multiply(beta_1, m) + tf.multiply(1.0 - beta_1, grad)
            next_v = tf.multiply(beta_2, v) + tf.multiply(1.0 - beta_2, grad_squared)

            update = next_m / (tf.sqrt(next_v) + epsilon)

            # Just adding the square of the weights to the loss function is *not*
            # the correct way of using L2 regularization/weight decay with Adam,
            # since that will interact with the m and v parameters in strange ways.
            #
            # Instead we want ot decay the weights in a manner that doesn't interact
            # with the m/v parameters. This is equivalent to adding the square
            # of the weights to the loss with plain (non-momentum) SGD.
            if weight_decay_rate > 0:
                update += weight_decay_rate * param

            update_with_lr = learning_rate * update

            next_param = param - update_with_lr

            if self.use_bfloat16_adam:
                next_m = tf.cast(next_m, dtype=tf.bfloat16)
                next_v = _encode_v(next_v)

            assignments.extend(
                [param.assign(next_param),
                 stored_m.assign(next_m),
                 stored_v.assign(next_v)])
        return tf.group(*assignments, name=name)

def reduce_rms(x):
    return tf.sqrt(tf.reduce_mean(tf.square(x)))
