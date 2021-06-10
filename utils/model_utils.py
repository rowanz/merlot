# Original work Copyright 2018 The Google AI Language Team Authors.
# Modified work Copyright 2019 Rowan Zellers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the Liense is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
import math
import re

import numpy as np
import six
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.contrib.tpu.python.ops import tpu_ops
from tensorflow.contrib.tpu.python.tpu import tpu_function
from tensorflow.compiler.tf2xla.python import xla


def assert_rank(tensor, expected_rank, name=None):
    """Raises an exception if the tensor rank is not of the expected rank.

    Args:
      tensor: A tf.Tensor to check the rank of.
      expected_rank: Python integer or list of integers, expected rank.
      name: Optional name of the tensor for the error message.

    Raises:
      ValueError: If the expected shape doesn't match the actual shape.
    """
    if name is None and not tf.executing_eagerly():
        name = tensor.name

    expected_rank_dict = {}
    if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True

    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank_dict:
        scope_name = tf.get_variable_scope().name
        raise ValueError(
            "For the tensor `%s` in scope `%s`, the actual rank "
            "`%d` (shape = %s) is not equal to the expected rank `%s`" %
            (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))


def get_shape_list(tensor, expected_rank=None, name=None):
    """Returns a list of the shape of tensor, preferring static dimensions.

    Args:
      tensor: A tf.Tensor object to find the shape of.
      expected_rank: (optional) int. The expected rank of `tensor`. If this is
        specified and the `tensor` has a different rank, and exception will be
        thrown.
      name: Optional name of the tensor for the error message.

    Returns:
      A list of dimensions of the shape of tensor. All static dimensions will
      be returned as python integers, and dynamic dimensions will be returned
      as tf.Tensor scalars.
    """
    if name is None and not tf.executing_eagerly():
        name = tensor.name

    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)

    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape


def gelu(input_tensor):
    """Gaussian Error Linear Unit.

    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415

    Args:
      input_tensor: float Tensor to perform activation.

    Returns:
      `input_tensor` with the GELU activation applied.
    """
    # math.sqrt needed for bfloat16 compatibility
    cdf = 0.5 * (1.0 + tf.erf(input_tensor / math.sqrt(2.0)))
    return input_tensor * cdf


def layer_norm(input_tensor, name=None, epsilon=1e-5):
    """Run layer normalization on the last dimension of the tensor."""
    name2use = f'LayerNorm_{name}' if name is not None else name
    with tf.variable_scope(name2use, default_name='LayerNorm'):
        dim = input_tensor.shape[-1].value
        gamma = tf.get_variable('gamma', [dim], initializer=tf.constant_initializer(1))
        beta = tf.get_variable('beta', [dim], initializer=tf.constant_initializer(0))

        cast_up_to_float32 = input_tensor.dtype == tf.bfloat16
        if cast_up_to_float32:
            input_tensor = tf.cast(input_tensor, dtype=tf.float32)

        mean, variance = tf.nn.moments(input_tensor, -1, keep_dims=True)
        scale_factor = tf.rsqrt(variance + epsilon) * gamma
        input_tensor = input_tensor * scale_factor - mean * scale_factor + beta
        if cast_up_to_float32:
            input_tensor = tf.cast(input_tensor, dtype=tf.bfloat16)
    return input_tensor


def group_norm(inputs,
               channels_axis=-1,
               num_groups=32,
               channels_per_group=None,
               epsilon=1e-5,
               mean_close_to_zero=True,
               name=None):
    """
    A better implementation of groupnorm
    :param inputs: n-dimensional inputs
    :param channels_axis. Which channel has the groups. All the other channels except channel 0 are considered
           reduction axes.
    :param mean_close_to_zero: The mean of `input` before ReLU will be close to zero
        when batch size >= 4k for Resnet-50 on TPU. If `True`, use
        `nn.sufficient_statistics` and `nn.normalize_moments` to calculate the
        variance. This is the same behavior as `fused` equals `True` in batch
        normalization. If `False`, use `nn.moments` to calculate the variance.
        When `mean` is close to zero, like 1e-4, use `mean` to calculate the
        variance may have poor result due to repeated roundoff error and
        denormalization in `mean`.  When `mean` is large, like 1e2,
        sum(`input`^2) is so large that only the high-order digits of the elements
        are being accumulated. Thus, use sum(`input` - `mean`)^2/n to calculate
        the variance has better accuracy compared to (sum(`input`^2)/n - `mean`^2)
        when `mean` is large.
    :return:
    """
    name2use = f'GroupNorm_{name}' if name is not None else name
    with tf.variable_scope(name2use, default_name='GroupNorm'):
        x_shape = get_shape_list(inputs)

        # Make it positive for convenience
        channels_axis = len(x_shape) + channels_axis if channels_axis < 0 else channels_axis

        # Reshape into groups
        channels = x_shape[channels_axis]
        if num_groups is None:
            assert channels_per_group is not None
            num_groups = channels // channels_per_group
        elif channels_per_group is None:
            channels_per_group = channels // num_groups
        else:
            if channels != num_groups * channels_per_group:
                raise ValueError("Num groups = {} channels per group = {} but channels = {}".format(
                    num_groups, channels_per_group, channels
                ))
        if channels % channels_per_group != 0:
            raise ValueError('%d channels is not commensurate with %d channels/gp.' %
                             (channels, channels_per_group))

        axes_before_channels = list(x_shape[:channels_axis])
        axes_after_channels = list(x_shape[channels_axis + 1:])
        new_shape = axes_before_channels + [num_groups, channels_per_group] + axes_after_channels
        x_reshape = tf.reshape(inputs, new_shape)

        # Cast up to float32 if it was originally float16
        cast_up_to_float32 = x_reshape.dtype == tf.bfloat16
        if cast_up_to_float32:
            x_reshape = tf.cast(x_reshape, tf.float32)

        # Determine the dimensions across which moments are calculated. Skip batch axis.
        moments_axes = [a + 1 if a >= channels_axis else a for a in range(1, len(x_shape))]

        # Calculate the moments.
        if mean_close_to_zero:
            # One pass algorithm returns better result when mean is close to zero.
            counts, means_ss, variance_ss, _ = tf.nn.sufficient_statistics(
                x_reshape, moments_axes, keep_dims=True)
            mean, variance = tf.nn.normalize_moments(
                counts, means_ss, variance_ss, shift=None)
        else:
            mean, variance = tf.nn.moments(x_reshape, moments_axes, keep_dims=True)

        x_normed = (x_reshape - mean) * tf.math.rsqrt(variance + epsilon)

        # This matches the shape of X
        params_shape_broadcast = ([1] * len(axes_before_channels) +
                                  [channels] +
                                  [1] * len(axes_after_channels))

        gammas = tf.get_variable(name='gamma', shape=[channels], initializer=tf.constant_initializer(1),
                                 dtype=tf.float32)
        gammas = tf.reshape(gammas, params_shape_broadcast)

        betas = tf.get_variable(name='beta', shape=[channels], initializer=tf.zeros_initializer(), dtype=tf.float32)
        betas = tf.reshape(betas, params_shape_broadcast)

        outputs = tf.reshape(x_normed, x_shape) * gammas + betas
        if cast_up_to_float32:
            return tf.cast(outputs, tf.bfloat16)
        return outputs


def one_hot_gather(x, idx):
    """
    Does a one-hot gather on a single axis, 0
    :param x: [N, H] tensor with a float dtype
    :param idx: 1 dimensional int32 with indices 0...N
    :return:
    """
    N, H = get_shape_list(x, 2)
    get_shape_list(idx, 1)
    idx_oh = tf.one_hot(idx, depth=N, dtype=tf.bfloat16 if x.dtype == tf.bfloat16 else tf.float32)
    return tf.matmul(idx_oh, x)


def embedder(x, name, vocab_size, embedding_size, initializer_range=0.02,
             use_one_hot_embeddings=True):
    """
    Helper function for creating embeddings on TPUs
    :param x: Input to be used
    :param name: What to call it
    :param vocab_size:
    :param embedding_size:
    :param initializer_range: Will be a truncated normal in this range
    :return:
    """

    embedding_table = tf.get_variable(
        name=name,
        shape=[vocab_size, embedding_size],
        initializer=tf.truncated_normal_initializer(stddev=initializer_range),
    )

    less_than_max = tf.assert_less_equal(tf.reduce_max(x), vocab_size - 1)
    gt_zero = tf.assert_greater_equal(tf.reduce_min(x), 0)
    with tf.control_dependencies([less_than_max, gt_zero]):
        if use_one_hot_embeddings:
            output_flat = one_hot_gather(embedding_table, idx=tf.reshape(x, [-1]))
            embedded_input = tf.reshape(output_flat, get_shape_list(x) + [embedding_size])
        else:
            embedded_input = tf.nn.embedding_lookup(embedding_table, x)

    return embedded_input, embedding_table


def position_embedder(seq_length, name, max_position_embeddings, embedding_size, offset=0,
                      initializer_range=0.02):
    """

    :param seq_length: Length of the sequence to position embed. Must be less than max_position_embeddings.
    :param name: Name of the embedding
    :param max_position_embeddings: Highest it'll go
    :param embedding_size: dimension to map to
    :param offset: Currently this isn't supported but it's so you can deal with caching. In that case
                   we don't want to run all the old sequences through the transformer
    :param initializer_range: for truncated normal initializer
    :return:
    """
    # Do something special for position embeddings
    assert_op = tf.assert_less_equal(seq_length, max_position_embeddings)
    with tf.control_dependencies([assert_op]):
        full_position_embeddings = tf.get_variable(
            name=name,
            shape=[max_position_embeddings, embedding_size],
            initializer=tf.truncated_normal_initializer(stddev=initializer_range),
        )

        # Since the position embedding table is a learned variable, we create it
        # using a (long) sequence length `max_position_embeddings`. The actual
        # sequence length might be shorter than this, for faster training of
        # tasks that do not have long sequences.
        #
        # So `full_position_embeddings` is effectively an embedding table
        # for position [0, 1, 2, ..., max_position_embeddings-1], and the current
        # sequence has positions [0, 1, 2, ... seq_length-1], so we can just
        # perform a slice.
        if offset == 0:
            position_embeddings = tf.slice(full_position_embeddings, [0, 0], [seq_length, -1])[None]
        else:
            # Tensorflow is too stupid to allow slicing
            flat_pos_ids = (tf.range(seq_length, dtype=tf.int32) + offset)
            one_hot_pos_ids = tf.one_hot(flat_pos_ids, depth=max_position_embeddings)

            # [seq_length, full_position_embeddings], [full_position_embeddings, dim]
            seq_embeds = tf.matmul(one_hot_pos_ids, full_position_embeddings)
            position_embeddings = seq_embeds[None]

    return position_embeddings, full_position_embeddings


def raw_cross_entropy_with_logits(logits, labels, cls_level_weights=None):
    """
    :param logits: [..., num_classes]
    :param labels: [...]
    :param scores: [...] with scores of the labels. Optional
    :param cls_level_weights: [num_classes]. Weights per class
    :return: The raw loss of size [...]
    """
    # cls pred loss. do something a bit different since we have scores now
    num_classes = get_shape_list(logits)[-1]

    one_hot_labels = tf.one_hot(labels, depth=num_classes, dtype=logits.dtype)
    if cls_level_weights is not None:
        ndim = len(get_shape_list(labels))
        one_hot_labels *= cls_level_weights[tuple([None] * ndim)]

    cls_logprobs = tf.nn.log_softmax(logits, axis=-1)

    raw_loss = -tf.reduce_sum(cls_logprobs * one_hot_labels, axis=-1)
    return raw_loss


def dropout(input_tensor, dropout_prob):
    """Perform dropout.

    Args:
      input_tensor: float Tensor.
      dropout_prob: Python float. The probability of dropping out a value (NOT of
        *keeping* a dimension as in `tf.nn.dropout`).

    Returns:
      A version of `input_tensor` with dropout applied.
    """
    if dropout_prob is None or dropout_prob == 0.0:
        return input_tensor
    output = tf.nn.dropout(input_tensor, rate=dropout_prob)
    return output


def get_ltr_attention_mask(nd, ns, dtype):
    """
    this is a TPU compatible version of tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd)
    where the lower right triangle contains 1s
    """
    i = tf.range(nd)[:, None]
    j = tf.range(ns)
    m = i >= j - ns + nd
    return tf.cast(m, dtype)


def create_unilm_attention_mask(is_bidirectional, is_padding=None):
    """
    Creates a hybrid left-to-right as well as bidirectional attention mask
    :param is_bidirectional: [batch_size, seq_length] that's 1 if bidirectional otherwise 0
    :return: A float tensor that's [batch_size, from_seq_length, to_seq_length].

    Information flows from to_seq to from_seq.

    amask[b,i,j] = 1.0 if i >= j, or, if is_bidirectional[b,j]

    If we were doing caching (which we aren't) then from_seq_length could be a bit longer as we could maybe attend
    to the cache items.

    NOTE: The semantics of this are the same as OpenAI's masking from left to right fn.
    """
    batch_size, seq_length = get_shape_list(is_bidirectional, expected_rank=2)

    ltr_attention_mask = tf.range(seq_length)[:, None] >= tf.range(seq_length)
    joint_attention_mask = tf.cast(is_bidirectional[:, None, :], tf.bool) | ltr_attention_mask[None]

    if is_padding is not None:
        joint_attention_mask = tf.logical_and(joint_attention_mask, tf.math.logical_not(is_padding)[:, None])
    return tf.cast(joint_attention_mask, dtype=tf.float32)


def get_assignment_map_from_checkpoint(tvars, init_checkpoint, reference_name_transform=None):
    """Compute the union of the current variables and checkpoint variables."""
    assignment_map = {}
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)

    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        rhs_name = name if reference_name_transform is None else reference_name_transform(name)

        if rhs_name not in name_to_variable:
            continue
        assignment_map[name] = rhs_name
        initialized_variable_names[rhs_name] = 1
        initialized_variable_names[rhs_name + ":0"] = 1
    return (assignment_map, initialized_variable_names)


def construct_scalar_host_call(metric_dict, model_dir, prefix=""):
    """Construct a host call to log scalars when training on TPU.

    Args:
      metric_dict: A dict of the tensors to be logged.
      model_dir: The location to write the summary.
      prefix: The prefix (if any) to prepend to the metric names.

    Returns:
      A tuple of (function, args_to_be_passed_to_said_function)
    """
    import warnings

    warnings.warn("construct_scalar_host_call is deprecated. Use construct_host_call instead, it's better",
                  DeprecationWarning)

    metric_names = list(metric_dict.keys())

    def host_call_fn(global_step, *args):
        """Training host call. Creates scalar summaries for training metrics.

        This function is executed on the CPU and should not directly reference
        any Tensors in the rest of the `model_fn`. To pass Tensors from the
        model to the `metric_fn`, provide as part of the `host_call`. See
        https://www.tensorflow.org/api_docs/python/tf/contrib/tpu/TPUEstimatorSpec
        for more information.

        Arguments should match the list of `Tensor` objects passed as the second
        element in the tuple passed to `host_call`.

        Args:
          global_step: `Tensor with shape `[batch]` for the global_step
          *args: Remaining tensors to log.

        Returns:
          List of summary ops to run on the CPU host.
        """
        step = global_step[0]
        with tf.contrib.summary.create_file_writer(
                logdir=model_dir, filename_suffix=".host_call").as_default():
            with tf.contrib.summary.always_record_summaries():
                for i, name in enumerate(metric_names):
                    tf.contrib.summary.scalar(prefix + name, args[i][0], step=step)

                return tf.contrib.summary.all_summary_ops()

    # To log the current learning rate, and gradient norm for Tensorboard, the
    # summary op needs to be run on the host CPU via host_call. host_call
    # expects [batch_size, ...] Tensors, thus reshape to introduce a batch
    # dimension. These Tensors are implicitly concatenated to
    # [params['batch_size']].
    global_step_tensor = tf.reshape(
        tf.compat.v1.train.get_or_create_global_step(), [1])
    other_tensors = [tf.reshape(metric_dict[key], [1]) for key in metric_names]

    return host_call_fn, [global_step_tensor] + other_tensors


def construct_host_call(scalars_to_log, model_dir, iterations_per_loop=100):
    """
    Constructs the host call function + arguments for logging. You can plug this directly into the TF Estimator
    :param scalars_to_log: {name: scalar} tensor that will be logged.
    :param model_dir: Where to put everything
    :param iterations_per_loop: How long to flush
    :return:
    """

    def host_call_fn(global_step, **kwargs):
        """Training host call. Creates scalar summaries for training metrics.

        This function is executed on the CPU and should not directly reference
        any Tensors in the rest of the `model_fn`. To pass Tensors from the
        model to the `metric_fn`, provide as part of the `host_call`. See
        https://www.tensorflow.org/api_docs/python/tf/contrib/tpu/TPUEstimatorSpec
        for more information.

        Arguments should match the list of `Tensor` objects passed as the second
        element in the tuple passed to `host_call`.

        Returns:
          List of summary ops to run on the CPU host.
        """
        # Outfeed supports int32 but global_step is expected to be int64.
        global_step = tf.reduce_mean(global_step)
        # Host call fns are executed FLAGS.iterations_per_loop times after one
        # TPU loop is finished, setting max_queue value to the same as number of
        # iterations will make the summary writer only flush the data to storage
        # once per loop.
        with tf.contrib.summary.create_file_writer(model_dir, max_queue=iterations_per_loop).as_default():
            with tf.contrib.summary.always_record_summaries():
                for k, v in sorted(kwargs.items(), key=lambda x: (len(x[0].split('/')), x[0])):
                    tf.contrib.summary.scalar(
                        k, tf.reduce_mean(v), step=global_step)
                return tf.contrib.summary.all_summary_ops()

    # To log the loss, current learning rate, and epoch for Tensorboard, the
    # summary op needs to be run on the host CPU via host_call. host_call
    # expects [batch_size, ...] Tensors, thus reshape to introduce a batch
    # dimension. These Tensors are implicitly concatenated to
    # [params['batch_size']].
    host_call_dict = {name: scalar[None] for name, scalar in scalars_to_log.items()}
    host_call_dict['global_step'] = tf.reshape(tf.compat.v1.train.get_or_create_global_step(), [1])

    return host_call_fn, host_call_dict


def pad_to_fixed_size(data, pad_value, output_shape, axis=0,
                      truncate=True,
                      name=None):
    """
    Pads the data to be a fixed size in the dimensions specified by axis.

    :param data: n-dimensional input.
    :param pad_value: What we will pad with
    :param output_shape: The desired output shape. This has to cover everything, not just axis.
    :param truncate: If True (default), we will TRUNCATE in the dimensions specifed by axis if we're over.
    :param axis: The axes to pad in. Pass a list to pad multiple dims.
    :return:
    """
    with tf.name_scope(name, default_name='pad_to_fixed_size', values=output_shape):
        axes = [axis] if isinstance(axis, int) else axis

        # Truncate if too long.
        pad_data = tf.identity(data)
        if truncate:
            slice_obj = [slice(0, os_i if i in axes else None, None) for i, os_i in enumerate(output_shape)]
            pad_data = pad_data[tuple(slice_obj)]

        # Anything not being padded, we assume is the output shape.
        current_shape = get_shape_list(pad_data, expected_rank=len(output_shape))
        for i, os_i in enumerate(output_shape):
            if i not in axes:
                current_shape[i] = os_i

        asserts = []
        for ax in axes:
            asserts.append(
                tf.Assert(tf.less_equal(current_shape[ax], output_shape[ax]), [current_shape[ax], output_shape[ax], ax])
            )

        with tf.control_dependencies(asserts):
            for ax in axes:
                pad_length = output_shape[ax] - current_shape[ax]
                pad_shape = [pad_length if i == ax else cs_i
                             for i, cs_i in enumerate(current_shape)]

                paddings = pad_value * tf.ones(pad_shape, dtype=data.dtype)
                pad_data = tf.concat([pad_data, paddings], axis=ax)

                # Update the dimension we padded in
                current_shape[ax] = output_shape[ax]

        pad_data = tf.reshape(pad_data, output_shape)
        return pad_data


def bfloat16_getter():
    """
    This is the magic that you need in order to get bfloat16 to work without messing up everything
    
    usually if you use bfloat16_scope that changes the variable scopes. but instead you can do
      with variable_scope.variable_scope(
      '', custom_getter=bfloat16_scope()) as varscope:
    
    :return: the getter
    """

    def inner_custom_getter(getter, *args, **kwargs):
        """Custom getter that forces variables to have type self.variable_type."""
        cast_to_bfloat16 = False
        requested_dtype = kwargs['dtype']
        if requested_dtype == tf.bfloat16:
            # Only change the variable dtype if doing so does not decrease variable
            # precision.
            kwargs['dtype'] = tf.float32
            cast_to_bfloat16 = True
        var = getter(*args, **kwargs)
        # This if statement is needed to guard the cast, because batch norm
        # assigns directly to the return value of this custom getter. The cast
        # makes the return value not a variable so it cannot be assigned. Batch
        # norm variables are always in fp32 so this if statement is never
        # triggered for them.
        if cast_to_bfloat16:
            var = tf.cast(var, tf.bfloat16)
        return var

    return inner_custom_getter


def binomial_sample(n, p):
    """
    Sample from a binomial.

    {\displaystyle f(k,n,p)=\Pr(k;n,p)=\Pr(X=k)={\binom {n}{k}}p^{k}(1-p)^{n-k}}
    so the logs are given by
    log(n!) - log(k!) - log((n-k!)) + k*log(p) + (n-k)*log(1-p)

    :param n: the n term (int)
    :param p: the p term (float)
    :return:
    """
    with tf.name_scope('binomial_sample'):
        # We can drop the initial n! term becauuse thats a constant
        counts = tf.cast(tf.range(0, n + 1), dtype=tf.float32)
        n_float = tf.cast(n, dtype=tf.float32)

        logits = -tf.math.lgamma(1. + n_float - counts) - tf.math.lgamma(1. + counts) + counts * tf.math.log(p) + (
                n_float - counts) * tf.math.log1p(-p)
        res = tf.reshape(tf.random.categorical(logits[None], dtype=tf.int32, num_samples=1), [])
    return res


def encode_string(tf_string, string_len):
    """
    Encodes the string into something TPU-able

    :param tf_string: string
    :param string_len: length
    :return: an encoded thing
    """
    out_raw = tf.cast(tf.io.decode_raw(tf_string, out_type=tf.uint8), dtype=tf.int32)[:string_len]
    return pad_to_fixed_size(out_raw, 0, [string_len])


def random_categorical_without_replacement(logits, num_samples):
    """
    Courtesy of https://github.com/tensorflow/tensorflow/issues/9260#issuecomment-437875125
    :param logits: [N] logits that are unscaled log probabilities
    :param num_samples:  <= N
    :return: num_samples inds that don't have repeatz
    """
    z = -tf.log(-tf.log(tf.random.uniform(tf.shape(logits), 0, 1)))
    _, indices = tf.nn.top_k(logits + z, num_samples)
    return tf.cast(indices, dtype=tf.int32)


def decode_string(x):
    return ''.join([chr(c) for c in x.astype(np.uint8) if c != 0])


def validate_shape(tensor, desired_size):
    """
    Double checks that the tensor is the same shape as the desired size
    :param tensor: A tensor
    :param desired_size: A tuple of the desired size
    """
    tensor_size = get_shape_list(tensor, expected_rank=len(desired_size))
    for dim_a, dim_b in zip(tensor_size, desired_size):
        if dim_a is None:
            pass
        elif dim_b is None:
            pass
        elif dim_a != dim_b:
            raise ValueError("For tensor {}: Invalid shape {} vs the desired {}".format(
                tensor.name, tensor_size, desired_size))


def tpu_cross_replica_stack(tensor, num_groups=1):
    """
    Replicates across groups, adapted from code in simclr
    :param tensor:
    :param num_groups:
    :return:
    """
    tpu_context = tpu_function.get_tpu_context()
    num_shards = tpu_context.number_of_shards
    if num_shards is None or num_shards <= 1:
        return tensor[None], 0

    num_shards_per_group = num_shards // num_groups
    if num_shards_per_group == 0:
        raise ValueError(
            f"Invalid num_shards_per_group={num_shards_per_group} for tpu_cross_replica_stack. num_shards={num_shards}, num_groups={num_groups}")
    group_assignment = [[
        x for x in range(num_shards) if x // num_shards_per_group == y
    ] for y in range(num_groups)]
    # Sounds weird but this is because otherwise it turns into an "s32" - what even is an s32?@!?!?
    my_group_idx = tf.mod(tf.cast(tf.cast(xla.replica_id(), tf.uint32), tf.int32), num_shards_per_group)
    with tf.name_scope('tpu_cross_replica_concat'):
        # This creates a tensor that is like the input tensor but has an added
        # replica dimension as the outermost dimension. On each replica it will
        # contain the local values and zeros for all other values that need to be
        # fetched from other replicas.
        ext_tensor = tf.scatter_nd(
            indices=[[my_group_idx]],
            updates=[tensor],
            shape=[num_shards_per_group] + tensor.shape.as_list())

        # As every value is only present on one replica and 0 in all others, adding
        # them all together will result in the full tensor on all replicas.
        ext_tensor = tf.tpu.cross_replica_sum(ext_tensor, group_assignment)
        return ext_tensor, my_group_idx


def position_embedder2d(num_h, num_w, embedding_size, name='pos_embs', num_img=1, max_position_embeddings=64,
                        max_nimg=4, num_cls_emb=1, initializer_range=0.02):
    """
    This is the same as a 2D pos emb BUT easier to change?
    :param num_h:
    :param num_w:
    :param embedding_size:
    :param name:
    :param num_img:
    :param max_position_embeddings:
    :param max_nimg:
    :param initializer_range:
    :return: [num_img * (1 + num_h * num_w), embedding_size] pos emb
    """
    with tf.variable_scope(name):
        pos_embs_3d = tf.get_variable(
            name='pos_embs',
            shape=[max_nimg, max_position_embeddings, max_position_embeddings, embedding_size],
            initializer=tf.truncated_normal_initializer(stddev=initializer_range),
        )
        cls_embs = tf.get_variable(
            name='cls_emb',
            shape=[max_nimg, num_cls_emb, embedding_size],
            initializer=tf.truncated_normal_initializer(stddev=initializer_range),
        ) if num_cls_emb > 0 else None

        full_pe = tf.reshape(pos_embs_3d[:num_img, :num_h, :num_w], [num_img, num_h * num_w, embedding_size])
        if num_cls_emb > 0:
            full_pe = tf.concat([cls_embs[:num_img], full_pe], 1)
        return tf.reshape(full_pe, [num_img * (num_cls_emb + num_h * num_w), embedding_size])


def sample_bernoulli(p_a):
    is_a = tf.random.categorical(tf.math.log([[1.0 - p_a, p_a]]), dtype=tf.int32, num_samples=1)
    is_a = tf.cast(tf.reshape(is_a, []), dtype=tf.bool)
    return is_a


def sample_bernoullis(p_a, shape):
    shape_np = np.array(shape)
    assert np.all(shape_np >= 1)
    assert shape_np.size > 0
    size = int(np.prod(shape))
    is_a = tf.random.categorical(tf.math.log([[1.0 - p_a, p_a]]), dtype=tf.int32, num_samples=size)
    is_a = tf.cast(tf.reshape(is_a, shape), dtype=tf.bool)
    return is_a


def lightweight_image_augment(image, strength=0.4, augment_prob=0.5,
                              allowed_transforms='all'):
    """
    augmentations from simclr paper
    :param image: [h, w, 3] -- float32 in the range of 0.0 to 1.0
    :param strength: magic number from simclr. i made hue weaker
    :param augment_prob: probability of doing an augmentation at all
    :return: image augmented
    """
    max_brightness_delta = 0.8 * strength
    max_contrast_delta = 0.8 * strength
    max_saturation_delta = 0.8 * strength
    max_hue_delta = 0.1 * strength

    def _brightness_transform(x):
        with tf.name_scope('brightness'):
            brightness_factor = tf.random_uniform([1, 1, 3], 1.0 - max_brightness_delta,
                                                  1.0 + max_brightness_delta)
            return x * brightness_factor

    def _grayscale_transform(x):
        V = tf.ones([3, 3], dtype=tf.float32) / 3.0
        return tf.matmul(x, V)

    def _contrast_transform(x):
        with tf.name_scope('contrast'):
            contrast_factor = tf.random_uniform([1, 1, 3], 1.0 - max_contrast_delta, 1.0 + max_contrast_delta)
            x_mean = tf.reduce_mean(x, [0, 1], keep_dims=True)
            return (x - x_mean) * contrast_factor + x_mean

    def _hsb_transform(x, do_hue=True, do_saturation=True, do_brightness=True, do_grayscale=False):
        with tf.name_scope('HS'):

            # First multiply
            sat_mult_factor = tf.random_uniform([], 1.0 - max_saturation_delta, 1.0 + max_saturation_delta) \
                if do_saturation else tf.ones([], dtype=tf.float32)
            hue_mult_factor = tf.ones([], dtype=tf.float32)
            bri_mult_factor = tf.random_uniform([], 1.0 - max_brightness_delta,
                                                1.0 + max_brightness_delta) if do_brightness else tf.ones([1],
                                                                                                          dtype=tf.float32)

            if do_grayscale:
                # Grayscale 1/10th of the time
                color_scale = tf.cast(sample_bernoulli(p_a=0.9), dtype=tf.float32)
                hue_mult_factor *= color_scale
                sat_mult_factor *= color_scale

            mult_factor = tf.stack([hue_mult_factor, sat_mult_factor, bri_mult_factor], -1)

            hsv_img = tf.image.rgb_to_hsv(x) * mult_factor[None, None]

            if do_hue:
                hue_add_factor = tf.random_uniform([1], -max_hue_delta, max_hue_delta)
                add_factor = tf.concat([hue_add_factor, tf.zeros([2], dtype=tf.float32)], 0)
                hsv_img += add_factor[None, None]
            return tf.image.hsv_to_rgb(hsv_img)

    if allowed_transforms == 'all':
        transforms = [_brightness_transform, _contrast_transform, _hsb_transform, _grayscale_transform]
    else:
        name_to_transform = {'brightness': _brightness_transform, 'contrast': _contrast_transform,
                             'hsb': _hsb_transform, 'grayscale': _grayscale_transform}
        tf.logging.info("USING TRANSFORMS {}".format(allowed_transforms.split(',')))
        transforms = [name_to_transform[name] for name in allowed_transforms.split(',')]

    def apply_transform(x, i):
        x = tf.switch_case(i, [lambda: t(x) for t in transforms])
        return x

    augment_idx = tf.reshape(
        tf.random.categorical(tf.math.log([[1.0] * len(transforms)]), dtype=tf.int32, num_samples=1), [])
    with tf.name_scope('augment'):
        image = tf.cond(
            sample_bernoulli(augment_prob),
            lambda: tf.clip_by_value(apply_transform(image, augment_idx), clip_value_min=0.0, clip_value_max=1.0),
            lambda: image,
        )
    return image


def apply_with_random_selector(x, func, num_cases):
    """Computes func(x, sel), with sel sampled from [0...num_cases-1].
    Args:
      x: input Tensor.
      func: Python function to apply.
      num_cases: Python int32, number of cases to sample sel from.
    Returns:
      The result of func(x, sel), where func receives the value of the
      selector as a python integer, but sel is sampled dynamically.
    """
    sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
    # Pass the real x only to one of the func calls.
    return control_flow_ops.merge([
        func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
        for case in range(num_cases)])[0]


def resize_and_pad(image, desired_output_size,
                   random_scale_min=0.1, random_scale_max=2.0, do_random_scale=False,
                   resize_method=tf.image.ResizeMethod.BILINEAR):
    """


    :param image:
    :param desired_output_size:
    :param boxes:
    :param random_scale_min:
    :param random_scale_max:
    :param do_random_scale:
    :return:
    """
    desired_height, desired_width = desired_output_size
    desired_height_f = tf.cast(desired_height, dtype=tf.float32)
    desired_width_f = tf.cast(desired_width, dtype=tf.float32)

    height = tf.cast(tf.shape(image)[0], tf.float32)
    width = tf.cast(tf.shape(image)[1], tf.float32)

    if do_random_scale:
        random_scale_factor = tf.random_uniform([], random_scale_min, random_scale_max)
        scaled_y = tf.cast(random_scale_factor * desired_height_f, tf.int32)
        scaled_x = tf.cast(random_scale_factor * desired_width_f, tf.int32)

        # Recompute the accurate scale_factor using rounded scaled image size.
        image_scale_y = tf.cast(scaled_y, tf.float32) / height
        image_scale_x = tf.cast(scaled_x, tf.float32) / width
        image_scale = tf.minimum(image_scale_x, image_scale_y)

        # Conceptual captions has some REALLY WIDE images I believe
        # this ensures that we won't scale any side lower than to 64
        image_scale = tf.maximum(image_scale, 64.0 / tf.minimum(height, width))

        # Select non-zero random offset (x, y) if scaled image is larger than
        # self._output_size.
        scaled_height = tf.cast(height * image_scale, tf.int32)
        scaled_width = tf.cast(width * image_scale, tf.int32)
        offset_y = tf.cast(scaled_height - desired_height, tf.float32)
        offset_x = tf.cast(scaled_width - desired_width, tf.float32)
        offset_y = tf.maximum(0.0, offset_y) * tf.random_uniform([], 0, 1)
        offset_x = tf.maximum(0.0, offset_x) * tf.random_uniform([], 0, 1)
        offset_y = tf.cast(offset_y, tf.int32)
        offset_x = tf.cast(offset_x, tf.int32)
    else:
        image_scale_y = desired_height_f / height
        image_scale_x = desired_width_f / width
        image_scale = tf.minimum(image_scale_x, image_scale_y)
        scaled_height = tf.cast(height * image_scale, tf.int32)
        scaled_width = tf.cast(width * image_scale, tf.int32)
        offset_y = tf.constant(0)
        offset_x = tf.constant(0)

    # Now resize and crop
    if isinstance(resize_method, str) and resize_method == 'random' and do_random_scale:
        tf.logging.info("Random resize method!")
        image = apply_with_random_selector(
            image,
            lambda x, method: tf.image.resize_images(x, [scaled_height, scaled_width], method, align_corners=True),
            num_cases=4)
    elif isinstance(resize_method, str):
        tf.logging.info(f"you passed in {resize_method} but doing bilinear resize instead")
        image = tf.image.resize_images(image, [scaled_height, scaled_width],
                                       method=tf.image.ResizeMethod.BILINEAR, align_corners=True)
    else:
        image = tf.image.resize_images(image, [scaled_height, scaled_width], method=resize_method, align_corners=True)

    image = image[offset_y:offset_y + desired_height, offset_x:offset_x + desired_width, :]
    image = tf.image.pad_to_bounding_box(image, 0, 0, desired_height, desired_width)

    if isinstance(desired_height, int) and isinstance(desired_width, int):
        image.set_shape([desired_height, desired_width, 3])
    else:
        tf.logging.info("Cant set shape bc desired height/width are dynamic")

    effective_height = tf.minimum(scaled_height, desired_height)
    effective_width = tf.minimum(scaled_width, desired_width)

    image_info = tf.stack([
        tf.cast(effective_height, dtype=tf.float32),
        tf.cast(effective_width, dtype=tf.float32),
        1.0 / image_scale,
        height,
        width])
    return image, image_info
