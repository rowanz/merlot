import math

import tensorflow as tf

from utils.model_utils import get_shape_list, dropout, gelu, layer_norm, bfloat16_getter, position_embedder2d


def _attention_projection_and_transpose(x_flat, batch_size, seq_length, num_attention_heads, size_per_head,
                                        name, initializer_range=0.02):
    """
    :param x_flat: [batch_size*seq_length, width]
    :return: A fixed up tensor of size [batch_size, num_attention_heads, seq_length, size_per_head]
    """
    batch_size_seq_length, dim = get_shape_list(x_flat, expected_rank=2)

    if dim != size_per_head * num_attention_heads:
        raise ValueError("passed in a tensor of shape {} when size_per_head={} and num_attention_heads={}".format(
            (batch_size_seq_length, dim), size_per_head, num_attention_heads
        ))

    projected = tf.layers.dense(
        x_flat,
        num_attention_heads * size_per_head,
        name=name,
        kernel_initializer=create_initializer(initializer_range))

    projected = tf.reshape(
        projected, [batch_size, seq_length, num_attention_heads, size_per_head])
    output_tensor = tf.transpose(projected, [0, 2, 1, 3])
    return output_tensor


def attention_layer(x_flat, attention_mask, batch_size, seq_length, size_per_head=64, num_attention_heads=12, *,
                    x_enc_flat=None, enc_seq_length=None, cache=None,
                    initializer_range=0.02, hidden_dropout_prob=0.1,
                    attention_probs_dropout_prob=0.1,
                    return_3d_tensor=False, custom_proj_dim=None,
                    do_cache=False):
    """

    :param x_flat: Tensor input, should be [batch_size*seq_length, dim]
    :param attention_mask: Attention mask to use of size [batch_size, from_length, to_length]
    :param batch_size:
    :param seq_length:
    :param size_per_head: dim = size_per_head * num_attention_heads
    :param num_attention_heads:  dim = size_per_head * num_attention_heads
    :param x_enc_flat: Optionally a separate thing to encode to. if None (default) we just use x_flat
    :param enc_seq_length: encoder sequence length
    :param cache: Optionally some past (cached) things of size
                [batch, 2, heads, sequence, features], where 2 is [k, v]
    :param initializer_range:
    :param hidden_dropout_prob
    :param attention_probs_dropout_prob:
    :param custom_proj_dim: If not None then we project to a DIFFERENT DIMENSION
    :return: A new tensor of shape [batch_size, seq_length, dim]
    as well as a new cache "cached_keys_and_values" that will be of size
                                   [batch_size, 2, num_attention_heads, seq_length, dim]
    """
    batch_size_seq_length, dim = get_shape_list(x_flat, expected_rank=2)

    if dim != size_per_head * num_attention_heads:
        raise ValueError("passed in a tensor of shape {} when size_per_head={} and num_attention_heads={}".format(
            (batch_size_seq_length, dim), size_per_head, num_attention_heads
        ))

    # [ batch_size, num_attention_heads, seq_length, size_per_head]
    query = _attention_projection_and_transpose(x_flat, batch_size=batch_size, seq_length=seq_length,
                                                num_attention_heads=num_attention_heads, size_per_head=size_per_head,
                                                name='query_layer',
                                                initializer_range=initializer_range)

    enc_seq_length = enc_seq_length if x_enc_flat is not None else seq_length
    x_enc_flat = x_enc_flat if x_enc_flat is not None else x_flat
    key = _attention_projection_and_transpose(x_enc_flat, batch_size=batch_size, seq_length=enc_seq_length,
                                              num_attention_heads=num_attention_heads, size_per_head=size_per_head,
                                              name='key_layer',
                                              initializer_range=initializer_range)

    value = _attention_projection_and_transpose(x_enc_flat, batch_size=batch_size, seq_length=enc_seq_length,
                                                num_attention_heads=num_attention_heads, size_per_head=size_per_head,
                                                name='value_layer',
                                                initializer_range=initializer_range)
    if (cache is not None) or do_cache:
        # Add to Cache
        cached_keys_and_values = tf.stack([key, value], axis=1)
    else:
        cached_keys_and_values = None

    # Things that were relevant from the cache
    if cache is not None:
        pk, pv = tf.unstack(cache, axis=1)
        key = tf.concat([pk, key], axis=-2)
        value = tf.concat([pv, value], axis=-2)

    # Multiply [batch_size, num_attention_heads, seq_length, size_per_head] with
    #          [batch_size, num_attention_heads, size_per_head, seq_length+cached_length] ->
    #          [batch_size, num_attention_heads, seq_length, seq_length+cached_length]
    attention_scores = tf.matmul(query, key, transpose_b=True)
    attention_scores = tf.multiply(attention_scores,
                                   1.0 / math.sqrt(float(size_per_head)))

    # Deal with the masking
    if len(get_shape_list(attention_mask)) == 3:
        tf.logging.info("Broadcasting the attention mask along the num_attention_heads dimension")
        mask2use = attention_mask[:, None]
    else:
        mask2use = attention_mask

    attention_scores = attention_scores * mask2use - tf.cast(1e10, attention_scores.dtype) * (
            1 - mask2use)

    attention_probs = tf.nn.softmax(attention_scores)

    if attention_probs_dropout_prob > 0.0:
        attention_probs = dropout(attention_probs, attention_probs_dropout_prob)

    # Multiply [batch_size, num_attention_heads, seq_length, seq_length+cached_length] with
    #          [batch_size, num_attention_heads, seq_length+cached_length, size_per_head] ->
    #          [batch_size, num_attention_heads, seq_length, size_per_head] ->
    context_layer = tf.matmul(attention_probs, value)

    # `context_layer` = [batch_size, seq_length, num_attention_heads, size_per_head]
    context_layer = tf.transpose(context_layer, [0, 2, 1, 3])
    if return_3d_tensor:
        context_layer = tf.reshape(context_layer, [batch_size, seq_length, num_attention_heads * size_per_head])
    else:
        context_layer = tf.reshape(context_layer, [batch_size * seq_length, num_attention_heads * size_per_head])

    proj_dim = num_attention_heads * size_per_head if custom_proj_dim is None else custom_proj_dim
    context_layer_projected = tf.layers.dense(
        context_layer,
        proj_dim,
        kernel_initializer=create_initializer(initializer_range),
        name='context_projection_layer'
    )
    context_layer_projected = dropout(context_layer_projected, hidden_dropout_prob)

    return context_layer_projected, attention_probs, cached_keys_and_values


def mlp_block(x_norm, intermediate_size, initializer_range=0.02, hidden_dropout_prob=0.1):
    """
    :param x_norm The attention output. It should be [batch_size*seq_length, dim]
    :param intermediate_size: the hidden projection. By default this is the input_dim * 4.

    :return:
    """
    hidden_size = get_shape_list(x_norm, expected_rank=[2, 3])[-1]
    intermediate_output = tf.layers.dense(
        x_norm,
        intermediate_size,
        activation=gelu,
        kernel_initializer=create_initializer(initializer_range),
        name='intermediate',
    )

    output_for_residual = tf.layers.dense(
        intermediate_output,
        hidden_size,
        name='output',
        kernel_initializer=create_initializer(initializer_range))
    output_for_residual = dropout(output_for_residual, hidden_dropout_prob)
    return output_for_residual


def create_initializer(initializer_range=0.02):
    """Creates a `truncated_normal_initializer` with the given range."""
    return tf.truncated_normal_initializer(stddev=initializer_range)


def transformer(hidden_state, attention_mask, config,
                dec2dec_cache=None, return_all_hiddens=False, return_attn_probs=False, compress_attn=False,
                return_cache=False):
    """
    Do a standard transformer with pre-LN. this is slightly different from grover
    :param hidden_state: [batch_size, dec_seq_length, hidden_size]
    :param attention_mask: [batch_size, dec_seq_length, dec_seq_length]
    :param config: settings and shit like that
    :param dec2dec_cache:          [batch_size, num_layers, 2, num_heads, cache_length, num_features]
    :return:
    """
    batch_size, seq_length, hidden_size = get_shape_list(hidden_state, 3)

    # Keep things 2d, like BERT
    hidden_state = tf.reshape(hidden_state, [batch_size * seq_length, hidden_size])

    tf.logging.info(
        f"Doing encoder only (or decoder only) transformer with shape [{batch_size},{seq_length},{hidden_size}]. Config:\n{config}\n~~")

    self_attn_probs = []
    hidden_states_all = [hidden_state]
    new_kvs = []
    for layer_idx in range(config['num_hidden_layers']):
        with tf.variable_scope('layer{:02d}'.format(layer_idx)):
            attention_output, attn_probs, new_kv = attention_layer(
                layer_norm(hidden_state, name='attn_ln0'),
                attention_mask=attention_mask,
                batch_size=batch_size,
                seq_length=seq_length,
                size_per_head=config['hidden_size'] // config['num_attention_heads'],
                num_attention_heads=config['num_attention_heads'],
                initializer_range=config['initializer_range'],
                hidden_dropout_prob=config['hidden_dropout_prob'],
                attention_probs_dropout_prob=config['attention_probs_dropout_prob'],
                cache=dec2dec_cache[:, layer_idx] if dec2dec_cache is not None else None,
                do_cache=return_cache,
            )
            if compress_attn:  # sum along the num_heads dim
                attn_probs = tf.reduce_mean(attn_probs, 1)

            self_attn_probs.append(attn_probs)
            new_kvs.append(new_kv)

            hidden_state += attention_output

            mlp_output = mlp_block(layer_norm(hidden_state, name='mlp_ln0'),
                                   intermediate_size=config['intermediate_size'],
                                   hidden_dropout_prob=config['hidden_dropout_prob'],
                                   )
            hidden_state += mlp_output
            hidden_states_all.append(hidden_state)

    # Final LN
    hidden_state = layer_norm(hidden_state, name='ln_final')

    hidden_state_reshaped = tf.reshape(hidden_state, [batch_size, seq_length, hidden_size])

    out_dict = {
        '_hidden_state_flat': hidden_state,
        'hidden_state': hidden_state_reshaped,
    }
    if return_all_hiddens:
        out_dict['all_hidden_states'] = tf.reshape(tf.stack(hidden_states_all, 1),
                                                   [batch_size, seq_length, len(hidden_states_all),
                                                    config['hidden_size']])

    if return_attn_probs:
        out_dict['self_attn_probs'] = tf.stack(self_attn_probs, 1)

    if return_cache:
        new_cache = tf.stack(new_kvs, 1)
        if dec2dec_cache is None:
            out_dict['new_dec2dec_cache'] = new_cache
        else:
            out_dict['new_dec2dec_cache'] = tf.concat([dec2dec_cache, new_cache], axis=-2)

    return out_dict
