import tensorflow as tf

from utils.model_utils import get_shape_list, dropout, gelu, layer_norm, bfloat16_getter, position_embedder2d, \
    group_norm
from utils.transformer import transformer, create_initializer


def fixed_padding(inputs, kernel_size):
    """
    Pads input along spatial dims
    :param inputs: [B, H, W, C]
    :param kernel_size: kernel to use
    :return: Padded [B, H + (k-1), W + (k-1), C]
    """
    assert kernel_size >= 1
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    return tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])


def batch_norm_relu(x, name=None, skip_relu=False):
    # Use group norm so model can't cheat
    x = group_norm(x, num_groups=32, channels_axis=-1, name=name, epsilon=1e-4, mean_close_to_zero=True)
    if skip_relu:
        return x
    return tf.nn.relu(x)


def conv2d_fixed_padding(inputs, filters, kernel_size, strides=1, weight_standardization=True):
    """
    Conv2d with padding that is independent of image size
    :param inputs:  [batch_size, h, w, c]
    :param filters: # filters
    :param kernel_size: kernel size
    :param strides: Int for strides
    :param weight_standardization: whether to use weight standardization -- do this if followed by a bn_relu
    :return:
    """
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size)

    conv0 = tf.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=('SAME' if strides == 1 else 'VALID'),
        data_format='channels_last',
        use_bias=False,
        kernel_initializer=tf.variance_scaling_initializer())

    batch_size, h, w, c = get_shape_list(inputs)
    conv0.build([batch_size, h, w, c])

    # Handle weight standardization
    kernel = conv0.weights[0]
    if weight_standardization:
        kernel = tf.cast(kernel, dtype=tf.float32)
        kernel_mean, kernel_variance = tf.nn.moments(kernel, [0, 1, 2], keep_dims=True)
        kernel = (kernel - kernel_mean) * tf.rsqrt(kernel_variance + 1e-5)

    if inputs.dtype == tf.bfloat16:
        kernel = tf.cast(kernel, dtype=tf.bfloat16)

    return tf.nn.conv2d(input=inputs, filter=kernel, strides=[1, strides, strides, 1], padding=conv0.padding.upper(),
                        data_format='NHWC', name='conv2d/Conv2D')


def bottleneck_block(inputs, filters, strides, use_projection=False):
    """
    :param inputs: [batch_size, h, w, c]
    :param filters: # filters for the first two convs (projection, and final use 4x as many)
    :param strides: Use avgpool with this kernel size
    :param use_projection: project, instead of use identity
    :return: [batch_size, h // strides, w // strides, 4 * filters]
    """
    shortcut = inputs
    if use_projection:
        shortcut = conv2d_fixed_padding(
            inputs=tf.nn.avg_pool2d(inputs, ksize=strides, strides=strides, padding='SAME',
                                    data_format='NHWC') if strides > 1 else inputs,
            filters=4 * filters,
            kernel_size=1)
        shortcut = batch_norm_relu(shortcut, skip_relu=True)

    inputs = conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=1)
    inputs = batch_norm_relu(inputs)

    inputs = conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=3)
    inputs = batch_norm_relu(inputs)

    if strides > 1:
        inputs = tf.nn.avg_pool2d(inputs, ksize=strides, strides=strides, padding='SAME', data_format='NHWC')
    inputs = conv2d_fixed_padding(inputs=inputs, filters=4 * filters, kernel_size=1)
    inputs = batch_norm_relu(inputs, skip_relu=True)
    return tf.nn.relu(inputs + shortcut)


def block_group(inputs, filters, blocks, strides, name):
    """

    :param inputs: [batch_size, h, w, c]
    :param filters: # filters for the first two convs (projection, and final use 4x as many)
    :param blocks: how many blocks
    :param strides: stride for first layer
    :param name: str name to use
    :return:
    """
    with tf.variable_scope(name):
        # Only the first block per block_group uses projection shortcut and strides.
        inputs = bottleneck_block(inputs, filters, strides=strides, use_projection=True)

        for _ in range(1, blocks):
            inputs = bottleneck_block(inputs, filters, strides=1)
        return tf.identity(inputs, f'{name}_final')


def lite_resnet50(inputs,
                  use_bfloat16=False,
                  num_resnet_layers=3,
                  width=64,
                  layers=(3, 4, 6, 3)):
    """
    My resnet

    * I also changed strides=1 there
    * Using weight standardization
    * Using GroupNorm

    :param inputs:
    :param use_bfloat16:
    :return:
    """
    with tf.variable_scope('resnet50lite', custom_getter=bfloat16_getter() if use_bfloat16 else None):
        tf.logging.info("LITE resnet50 with {} layers, bfloat16={}".format(num_resnet_layers, use_bfloat16))

        with tf.variable_scope('stem'):
            x0 = conv2d_fixed_padding(
                inputs=inputs,
                filters=width // 2,
                kernel_size=3,
                strides=2,
            )
            x0 = batch_norm_relu(x0, name='stem0')
            x1 = conv2d_fixed_padding(
                inputs=x0,
                filters=width // 2,
                kernel_size=3,
                strides=1,
            )
            x1 = batch_norm_relu(x1, name='stem1')
            x2 = conv2d_fixed_padding(
                inputs=x1,
                filters=width,
                kernel_size=3,
                strides=1,
            )
            x2 = batch_norm_relu(x2, name='stem2')
            c = tf.nn.avg_pool2d(x2, ksize=2, strides=2, padding='SAME', data_format='NHWC')

        for i in range(num_resnet_layers):
            tf.logging.info(f"Resnet layer c{i + 2} -> {layers[i]} blocks")
            c = block_group(
                inputs=c,
                filters=width * (2 ** i),
                blocks=layers[i],
                strides=1 if i == 0 else 2,
                name=f'block_group{i + 1}',
            )
        return c


def vision_transformer_backbone(image, model_config):
    """
    ViT backbone with or without resnet hybrid
    :param image:
    :param model_config:
    :return:
    """
    P = model_config['patch_size']
    use_bfloat16 = model_config['use_bfloat16']
    hidden_size = model_config['hidden_size']
    num_cls_emb = model_config.get('num_cls_emb', 2)
    resnet_layers = model_config.get('resnet_layers', [])
    tf.logging.info(f"P={P}\nuse_bfloat16={use_bfloat16}\nhidden_size={hidden_size}\nnum_cls_emb={num_cls_emb}\n"
                    f"resnet_layers={resnet_layers}")

    pseudo_batch_size, h0, w0, c = get_shape_list(image, 4)
    assert h0 % P == 0
    assert w0 % P == 0

    with tf.variable_scope('vision_transformer', custom_getter=bfloat16_getter() if use_bfloat16 else None):
        img_norm = image - 0.5
        if len(resnet_layers) == 0:
            tf.logging.info(f"Doing the initial {P}x{P} stem")
            x = tf.layers.conv2d(
                img_norm,
                filters=hidden_size,
                kernel_size=P,
                strides=P,
                padding='VALID',
                data_format='channels_last',
                use_bias=True,
                kernel_initializer=tf.variance_scaling_initializer(),
            )
        else:
            tf.logging.info(f"Doing resnet stem with {resnet_layers}")
            assert P == 16
            resnet_c = lite_resnet50(img_norm, use_bfloat16=use_bfloat16, num_resnet_layers=len(resnet_layers),
                                     layers=resnet_layers, width=64)

            # Convert between hidden sizes
            x = tf.layers.conv2d(
                resnet_c,
                filters=hidden_size,
                kernel_size=1,
                strides=1,
                padding='SAME',
                data_format='channels_last',
                use_bias=True,
                kernel_initializer=tf.variance_scaling_initializer(),
                name='conv_postresnet_proj',
            )

        h1 = h0 // P
        w1 = w0 // P
        num_patch = h1 * w1

        x = tf.reshape(x, [pseudo_batch_size, num_patch, hidden_size])
        x = tf.cast(x, dtype=tf.float32)
        x = tf.concat([tf.zeros([pseudo_batch_size, num_cls_emb, hidden_size], dtype=x.dtype), x], 1)
        pos_embs = position_embedder2d(num_cls_emb=num_cls_emb, num_h=h1, num_w=w1, num_img=1, max_nimg=1,
                                       embedding_size=hidden_size)
        x = layer_norm(x + pos_embs, name='ctx_patches_pre_ln')

        if model_config['use_bfloat16']:
            x = tf.cast(x, dtype=tf.bfloat16)

        attention_mask = tf.ones([pseudo_batch_size, num_patch + num_cls_emb, num_patch + num_cls_emb], dtype=x.dtype)
        vision_transformer_config = {k: v for k, v in model_config.items()}
        vision_transformer_config['num_hidden_layers'] = model_config.get('num_vision_transformer_hidden_layers',
                                                                          model_config['num_hidden_layers'])
        vision_transformer_config['hidden_dropout_prob'] = model_config.get('vit_hidden_dropout_prob',
                                                                            model_config['hidden_dropout_prob'])

        # We'll do a LN after the 2x2 pool
        vision_transformer_info = transformer(x, attention_mask, vision_transformer_config)
        tf.logging.info("ViT with {} layers".format(vision_transformer_config['num_hidden_layers']))

    # Now do the extraction
    vision_transformer_info['cls'] = vision_transformer_info['hidden_state'][:, :num_cls_emb]
    vision_transformer_info['seq'] = vision_transformer_info['hidden_state'][:, num_cls_emb:]

    # Avg pool to save memory
    if model_config['spatial_pool_size'] > 1:
        # Attention pool
        tf.logging.info("Resizing sequence of hidden states, currently of shape {}, using spatial_pool={}".format(
            get_shape_list(vision_transformer_info['hidden_state'], 3), model_config['spatial_pool_size']))

        seq_reshaped = tf.reshape(vision_transformer_info['seq'], [pseudo_batch_size, h1, w1, hidden_size])
        seq_reshaped = tf.nn.avg_pool2d(seq_reshaped, ksize=model_config['spatial_pool_size'],
                                        strides=model_config['spatial_pool_size'], padding='VALID', data_format='NHWC')
        h2 = h1 // model_config['spatial_pool_size']
        w2 = w1 // model_config['spatial_pool_size']

        vision_transformer_info['seq'] = tf.reshape(seq_reshaped, [pseudo_batch_size, h2 * w2, hidden_size])
        tf.logging.info(f"NEW size for seq: [batch {pseudo_batch_size}, size {h2} x {w2}, {hidden_size}]")
    else:
        h2 = h1
        w2 = w1

    vision_transformer_info['num_h'] = h2
    vision_transformer_info['num_w'] = w2
    return vision_transformer_info
