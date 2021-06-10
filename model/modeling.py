import copy

import tensorflow as tf

from utils import optimization
from utils.model_utils import get_assignment_map_from_checkpoint, \
    get_shape_list, layer_norm, construct_host_call, embedder, position_embedder, \
    dropout, tpu_cross_replica_stack, random_categorical_without_replacement, one_hot_gather, \
    raw_cross_entropy_with_logits, bfloat16_getter, gelu
from utils.neat_config import NeatConfig
from utils.transformer import create_initializer, transformer
from utils.vision_transformer import vision_transformer_backbone, position_embedder2d
import os
from utils.encode.encoder import get_encoder, MASK
import math


def project_and_norm(x, hidden_size, name='proj', add_intermediate=False, initializer_range=0.02):
    """
    :param x: 2d or 3d tensor
    :param hidden_size: Also use this for the contrastive dim
    :param name: Name of the projection
    :param add_intermediate:
    :param initializer_range:
    :return:
    """
    if add_intermediate:
        x = tf.layers.dense(
            x,
            hidden_size,
            kernel_initializer=create_initializer(initializer_range),
            name=f'{name}_intermediate',
            activation=gelu,
        )
        x = layer_norm(x, name=f'{name}_ln')

    x_proj = tf.layers.dense(
        x,
        hidden_size,
        kernel_initializer=create_initializer(initializer_range),
        name=name,
    )
    x_proj = tf.math.l2_normalize(x_proj, axis=-1)
    return x_proj


class MerlotModel(object):
    def __init__(self, config, is_training, use_tpu, image, input_ids, mask_input=False,
                 shuffled_idx_img=None,
                 img_mask=None, log_attention_probs=True):
        """

        :param config: The 'Model' component of a neatconfig
        :param is_training:  is_training: Training or testing mode
        :param use_tpu:
        :param image: This must be [batch_size * num_chunks, h, w, 3]. If you do something weird like transpose the image you
                      need to do that somewhere else
        :param input_ids: [batch_size, num_chunks, L]. OR PASS IN SOMETHING 2D AND num_chunks = 1
        :param input_ids_masked: [batch_size, num_chunks, L]. For training with mask LM
        :param video_src_ids: [batch_size, num_chunks] -- you don't have to worry about this unless you're doing video
                                                          it's just so we don't attend between videos for language-only
                                                          features
        :param shuffled_idx_img: [batch_size, num_chunks] <- for each segment, what should the position embedding offset be
        :param img_mask: [batch_size, num_chunks] is it valid
        :param log_attention_probs: Whether to log attention probs
        """
        self.config = copy.deepcopy(config)
        self.is_training = is_training
        self.use_tpu = use_tpu

        input_ids_shape = get_shape_list(input_ids, expected_rank=[2, 3])
        if len(input_ids_shape) == 2:
            tf.logging.info("input ids shape: {} setting num_chunks = 1".format(input_ids_shape))
            self.num_chunks = 1
            self.num_chunks_in_group = 1
            self.batch_size, self.lang_chunk_length = input_ids_shape
            self.input_ids = input_ids[:, None]
        else:
            self.input_ids = input_ids
            self.batch_size, self.num_chunks, self.lang_chunk_length = input_ids_shape
            self.num_chunks_in_group = self.config.get('num_chunks_in_group', self.num_chunks)
            assert self.num_chunks % self.num_chunks_in_group == 0

        self.num_imgs = self.config.get('num_imgs', 1)
        self.num_texts = self.config.get('num_texts', 1)
        self.img_batch_size = self.batch_size // self.num_texts

        if not is_training:
            self.config['hidden_dropout_prob'] = 0.0
            self.config['attention_probs_dropout_prob'] = 0.0

        self.encoder_pieces = []

        # Vision backbone on the entire image (batch_size * num_chunks) -- everything independently
        with tf.variable_scope('vision_backbone', custom_getter=self.custom_getter):
            self.vision_transformer_info = vision_transformer_backbone(image, self.config)

            # Target for contrastive loss
            self.img_trg_h = tf.cast(self.vision_transformer_info['cls'][:, 1], dtype=tf.float32)

            image_feats = tf.concat([
                self.vision_transformer_info['cls'][:, 0, None],
                self.vision_transformer_info['seq'],
            ], 1)
            image_feats = tf.cast(image_feats, dtype=tf.float32)
            if img_mask is None:
                img_mask = tf.ones([self.B // self.num_texts, self.num_imgs], dtype=tf.bool)
            else:
                img_mask = tf.reshape(img_mask, [self.B // self.num_texts, self.num_imgs])

            if (self.num_imgs > 1) or (self.num_texts > 1):
                image_feats = tf.reshape(image_feats, [self.img_batch_size, self.num_imgs, *image_feats.shape[1:]])
                tf.logging.info(f"Duplicating images / texts: num_imgs={self.num_imgs} and num_texts={self.num_texts}")
                if self.num_texts > 1:
                    # batch_size, num_imgs, P, hidden_size
                    image_feats = tf.reshape(tf.tile(image_feats[:, None], [1, self.num_texts, 1, 1, 1]),
                                             [self.B, *image_feats.shape[1:]])
                    img_mask = tf.reshape(tf.tile(img_mask[:, None], [1, self.num_texts, 1]),
                                          [self.B, *img_mask.shape[1:]])

            image_feats = tf.reshape(image_feats, [self.B, self.P * self.num_imgs, self.hidden_size])
            img_mask = tf.reshape(tf.tile(img_mask[:, :, None], [1, 1, self.P]), [self.B, self.P * self.num_imgs])

            # Add img-wise position embeddings so we can tell apart different images from one another
            image_feats += self.vision_pos_emb(shuffled_idx_img=shuffled_idx_img)
            image_feats = layer_norm(image_feats, name='final_ln')
            if self.use_bfloat16:
                image_feats = tf.cast(image_feats, dtype=tf.bfloat16)
            self.encoder_pieces.append({
                'name': 'viz',
                'x': image_feats,
                'is_valid': img_mask,
            })

        if mask_input:
            tf.logging.info("Masking input")
            self.lang_trg_h, self.lang_transformer_info = self.langonly_reps()
            self.lang_mask_info = self.mask_inputs()
            input_ids_to_use = self.lang_mask_info['masked_ids']
        else:
            input_ids_to_use = input_ids

        input_ids_to_use = tf.reshape(input_ids_to_use, [self.B, self.L])

        self.encoder_pieces.append({
            'name': 'lang',
            'x': self.embed_words(input_ids_to_use),
            'is_valid': tf.not_equal(input_ids_to_use, 0),
        })

        encoder_input = tf.concat([x['x'] for x in self.encoder_pieces], 1)
        is_valid = tf.concat([x['is_valid'] for x in self.encoder_pieces], 1)

        tf.logging.info("TRANSFORMER ON")
        for x in self.encoder_pieces:
            tf.logging.info("{}: {}".format(x['name'], get_shape_list(x['x'])))

        attention_mask = tf.logical_and(is_valid[:, None], is_valid[:, :, None])

        if self.config.get('disable_pairwise_lang_attn', False):
            tf.logging.info("Disabling pairwise language attention")
            segment_idx = tf.concat([tf.zeros(self.P, dtype=tf.int32),
                                     1 + tf.floor_div(tf.range(self.L), self.lang_chunk_length),
                                     ], 0)
            can_attend = tf.equal(segment_idx[:, None], segment_idx[None])
            can_attend = tf.logical_or(can_attend, tf.equal(segment_idx, 0)[None])
            can_attend = tf.logical_or(can_attend, tf.equal(segment_idx, 0)[:, None])
            attention_mask = tf.logical_and(attention_mask, can_attend)

        attention_mask = tf.cast(attention_mask, dtype=tf.bfloat16 if self.use_bfloat16 else tf.float32)
        with tf.variable_scope('encoder', custom_getter=self.custom_getter,
                               reuse=tf.AUTO_REUSE if self.config.get('share_params', True) else False):
            self.encoder_info = transformer(encoder_input, attention_mask, self.config,
                                            return_attn_probs=log_attention_probs, compress_attn=True)

        self.encoder_hidden_states = {}
        cur_len = 0
        for x in self.encoder_pieces:
            x['start'] = cur_len
            this_len = get_shape_list(x['x'])[1]
            x['end'] = cur_len + this_len
            cur_len = x['end']
            hs = self.encoder_info['hidden_state'][:, x['start']:x['end']]
            self.encoder_hidden_states[x['name']] = tf.cast(hs, dtype=tf.float32)  # Cast back to float32 in case

        if log_attention_probs:
            # [batch_size, L, L] (we just MEANed over the num layers dim]
            # Attention logging  (attention flows TO, attention comes FROM)
            # Need to cast to float32 for logging
            self_attn_probs = tf.cast(tf.reduce_mean(self.encoder_info['self_attn_probs'], 1), dtype=tf.float32)

            is_valid_f = tf.cast(is_valid, dtype=tf.float32)
            self_attn_probs *= is_valid_f[:, None] * is_valid_f[:, :, None]

            self_attn_probs = tf.reduce_mean(self_attn_probs, 0)  # Now it's [L, L]
            self_attn_probs /= tf.reduce_sum(self_attn_probs)

            attns = {}
            for x_to in self.encoder_pieces:
                for x_from in self.encoder_pieces:
                    attn_summ = self_attn_probs[x_to['start']:x_to['end'], x_from['start']:x_from['end']]
                    attns['{}2{}'.format(x_from['name'], x_to['name'])] = tf.reduce_sum(attn_summ)
            self.attention_log = {f'encoder/{k}': v for k, v in sorted(attns.items(), key=lambda x: x[0])}

    def lm_head(self, hidden_state):
        with tf.variable_scope("lm_head"):
            if self.config.get('do_projection', False):
                h0 = tf.layers.dense(
                    hidden_state,
                    self.hidden_size,
                    kernel_initializer=create_initializer(self.config['initializer_range']),
                    name='projection',
                    activation=gelu,
                )
                hidden_state = layer_norm(h0)

            logits = tf.matmul(hidden_state, self.word_embedding_table, transpose_b=True)
            if self.config.get('do_bias', False):
                output_bias = tf.get_variable(
                    "output_bias",
                    shape=[self.vocab_size],
                    initializer=tf.zeros_initializer())
                logits = tf.nn.bias_add(logits, output_bias)
        return logits

    @property
    def hidden_size(self):
        return self.config['hidden_size']

    @property
    def vocab_size(self):
        return self.config['vocab_size']

    @property
    def B(self):
        return self.batch_size * (self.num_chunks // self.num_chunks_in_group)

    @property
    def L(self):
        return self.lang_chunk_length * self.num_chunks_in_group

    @property
    def viz_chunk_length(self):
        return self.vision_transformer_info['num_h'] * self.vision_transformer_info['num_w'] + 1

    @property
    def P(self):
        return self.viz_chunk_length * self.num_chunks_in_group

    @property
    def dropout_prob(self):
        return self.config['hidden_dropout_prob']

    @property
    def use_bfloat16(self):
        return self.config['use_bfloat16']

    @property
    def custom_getter(self):
        return bfloat16_getter() if self.use_bfloat16 else None

    def embed_words(self, input_ids_2d, norm_scope_name='position_embeddings',
                    reuse=tf.AUTO_REUSE):
        """
        :param input_ids_2d:
        :param skip_pad: Whether to skip pad characters which occur in the middle (and might mean that certain
                         position embeddings like pos=31 never get seen)
        :param reuse:
        :return:
        """
        B, L = get_shape_list(input_ids_2d, expected_rank=2)
        tf.logging.info(f"embedding {B} x {L}")

        # Language
        with tf.variable_scope('word_embeddings', reuse=reuse):
            lang_emb, self.word_embedding_table = embedder(
                input_ids_2d,
                name='word_embeddings',
                vocab_size=self.vocab_size,
                embedding_size=self.hidden_size,
                initializer_range=self.config['initializer_range'],
                use_one_hot_embeddings=self.use_tpu,
            )

        with tf.variable_scope(norm_scope_name, reuse=reuse):
            pos_embs, pos_emb_table = position_embedder(
                L,
                name='position_embeddings',
                max_position_embeddings=self.config['max_position_embeddings'],
                embedding_size=self.hidden_size,
                initializer_range=self.config['initializer_range'],
            )
            emb_normed = layer_norm(lang_emb + pos_embs, name='embed_norm')
            emb_normed = dropout(emb_normed, dropout_prob=self.dropout_prob)
            if self.use_bfloat16:
                emb_normed = tf.cast(emb_normed, dtype=tf.bfloat16)
        return emb_normed

    def vision_pos_emb(self, shuffled_idx_img=None):
        """
        Add only image level position embeddings
        :param shuffled_idx_img: If not None, then we will do shuffle the input
        :return:
        """
        my_pe, img_pe_table = position_embedder(
            self.num_chunks_in_group * self.num_imgs,
            name='img_idx_pe',
            max_position_embeddings=self.config.get('max_vision_pos_embeddings', 1024),
            embedding_size=self.hidden_size,
            initializer_range=self.config['initializer_range'],
        )
        if shuffled_idx_img is None:
            tf.logging.info("NOT shuffling the vision input! this is probably what you want for downstream")
            my_pe = tf.tile(my_pe[:, :, None], [1, 1, self.viz_chunk_length, 1])
            my_pe = tf.reshape(my_pe, [1, self.P * self.num_imgs, self.hidden_size])
        else:
            tf.logging.info("!!!shuffling the vision input!!!!")
            # Idk how to handle these things
            assert self.num_imgs == 1
            assert self.num_texts == 1
            my_pe = one_hot_gather(img_pe_table, tf.reshape(shuffled_idx_img, [-1]))
            my_pe = tf.tile(my_pe[:, None], [1, self.viz_chunk_length, 1])
            my_pe = tf.reshape(my_pe, [self.B, self.P, self.hidden_size])

        # add extra position embeddings, since even though the vision transformer had position
        # embeddings we did an avgpool so they might have gotten washed out
        image_pe2d = position_embedder2d(num_h=self.vision_transformer_info['num_h'],
                                         num_w=self.vision_transformer_info['num_w'],
                                         embedding_size=self.hidden_size,
                                         num_img=1,
                                         num_cls_emb=1,
                                         max_nimg=1,
                                         initializer_range=self.config['initializer_range'],
                                         name='final_pe',
                                         )
        my_pe += tf.tile(image_pe2d, [self.num_chunks_in_group * self.num_imgs, 1])[None]
        return my_pe

    def langonly_reps(self):
        """
        Get language-only contrastive representations via a transformer
        :return:
        """
        # Put the entire sequence in, even if we wont use that later
        if 'langonly_num_chunks_in_group' in self.config:
            lang_nchunk_in_group = self.config['langonly_num_chunks_in_group']
            lang_ngroups = self.num_chunks // lang_nchunk_in_group
            assert lang_ngroups > 0
            assert self.num_chunks % lang_nchunk_in_group == 0
            tf.logging.info(f"Breaking up language only from {self.num_chunks} into ngroups={lang_ngroups} of size {lang_nchunk_in_group}")
            input_ids_2d = tf.reshape(self.input_ids, [self.batch_size * lang_ngroups, self.lang_chunk_length * lang_nchunk_in_group])
        else:
            input_ids_2d = tf.reshape(self.input_ids, [self.batch_size, self.lang_chunk_length * self.num_chunks])
        word_embs = self.embed_words(input_ids_2d, norm_scope_name='langonly_embeddings')

        use_bfloat16 = self.config['use_bfloat16']
        share_params = self.config.get('share_params', True)
        if share_params:
            tf.logging.info("Sharing parameters for lang-only decoder")

        with tf.variable_scope('encoder' if share_params else 'langonly_encoder',
                               custom_getter=bfloat16_getter() if use_bfloat16 else None):
            is_valid = tf.not_equal(input_ids_2d, 0)
            is_valid2d = tf.logical_and(is_valid[:, None], is_valid[:, :, None])
            attention_mask = tf.cast(is_valid2d, dtype=word_embs.dtype)

            lang_transformer_config = {k: v for k, v in self.config.items()}
            lang_transformer_config['num_hidden_layers'] = self.config['num_lang_transformer_hidden_layers']

            lang_transformer_info = transformer(word_embs, attention_mask, lang_transformer_config,
                                                return_attn_probs=True, compress_attn=True)
            hidden_state_pool = tf.reshape(
                lang_transformer_info['_hidden_state_flat'],
                [self.batch_size * self.num_chunks, self.lang_chunk_length, self.hidden_size],
            )[:, 0]

            if hidden_state_pool.dtype == tf.bfloat16:
                hidden_state_pool = tf.cast(hidden_state_pool, dtype=tf.float32)
        return hidden_state_pool, lang_transformer_info

    def mask_inputs(self):
        """
        return masked inputs -- can either use attention or not, and use spanbert masking or not
        :return:
        """
        input_ids_2d = tf.reshape(self.input_ids, [self.B, self.L])

        with tf.name_scope("masking"):
            # Default values: with probability 50% we draw from the top 20% of things
            topk_perc = self.config.get('masking_use_topk_from_attn_perc', 0.20)
            choose_topk_prob = self.config.get('masking_choose_topk_prob', 0.5)
            masking_rate = self.config.get('masking_rate', 0.2)
            do_spanbert = self.config.get('masking_do_spanbert', True)

            # Try to match t5, which has an expected value of 3
            # I get roughly the same when maxlen=4, doing symmetric, and p=0.35
            # These are the results
            spanbert_len_probs = self.config.get('masking_spanbert_len_probs', [0.625,0.25,0.125]) # saying EV = 2.0
            masking_use_attn = self.config.get('masking_use_attn', True)

            num_topk = int(self.L * topk_perc)
            num_to_mask = int(self.L * masking_rate)
            tf.logging.info(f"Masking~~~\n"
                            f"topk_perc:        {topk_perc}\n"
                            f"choose_topk_prob: {choose_topk_prob}\n"
                            f"masking_rate:     {masking_rate}\n"
                            f"do_spanbert:      {do_spanbert}\n"
                            f"masking_use_attn: {masking_use_attn}\n"
                            f"num_topk:         {num_topk}\n"
                            f"spanbert_len_probs{spanbert_len_probs}\n"
                            f"num_to_mask:      {num_to_mask} or {num_to_mask / self.L:.3f}\n")

            # for drawing a single one -- probability is
            #                                           topk_val * topk_perc
            # choose_topk_prob  =      ______________________________________________________
            #                          nontopk_val * (1.0 - topk_perc) + topk_val * topk_perc
            # means ->
            nontopk_val = 0.01
            topk_val = nontopk_val * choose_topk_prob * (1.0 - topk_perc) / (topk_perc * (1.0 - choose_topk_prob))

            #######################################################
            sentinel_idx = tf.range(self.L)
            is_special_token = tf.cast(tf.less(input_ids_2d, 100), dtype=tf.float32)

            if masking_use_attn:
                # Figure out which words are the most attended to
                # [B, src]
                attention_summs = tf.reduce_sum(self.lang_transformer_info['self_attn_probs'], [1, 2])
                # The language transformer might have used a different size
                attention_summs = tf.reshape(attention_summs, [self.B, self.L])
                attention_summs = tf.cast(attention_summs, dtype=tf.float32)
                # No masking out special tokens
                attention_summs *= (1.0 - is_special_token)

                _, attention_top_inds = tf.math.top_k(attention_summs, k=num_topk)
                is_important_oh = tf.reduce_any(tf.equal(attention_top_inds[..., None], sentinel_idx[None, None]), [1])
                mask_weight = tf.cast(is_important_oh, dtype=tf.float32) * (topk_val - nontopk_val) + nontopk_val
            else:
                mask_weight = tf.ones([self.B, self.L], dtype=tf.float32)

            # Don't mask out special tokens
            log_mask = tf.log(mask_weight) - 1e8 * is_special_token
            # Flip the order because spanbert favors masks that appear later in the list, and those had higher prob
            # of being added without replacement
            idx = random_categorical_without_replacement(log_mask, num_to_mask)[:, ::-1]

            if do_spanbert:
                # mask spans
                span_extra_lower = tf.reshape(tf.cast(tf.random.categorical(
                    tf.math.log(spanbert_len_probs)[None], num_samples=self.B * num_to_mask), dtype=tf.int32),
                    [self.B, num_to_mask])
                span_extra_upper = tf.reshape(tf.cast(tf.random.categorical(
                    tf.math.log(spanbert_len_probs)[None], num_samples=self.B * num_to_mask), dtype=tf.int32),
                    [self.B, num_to_mask])

                # [B, num_to_mask, L] basically YES if L is masked by this span
                span_start = idx - span_extra_lower
                span_end = idx + span_extra_upper

                # Subtly this means we will never actually do anything with span 0, that's fine I guess
                does_match = tf.logical_and(
                    tf.greater_equal(sentinel_idx[None, None], span_start[..., None]),
                    tf.less_equal(sentinel_idx[None, None], span_end[..., None]),
                )
                which_match = tf.cast(tf.argmax(tf.cast(does_match, dtype=tf.float32), 1), dtype=tf.float32)
                which_match *= (1.0 - is_special_token)
                # Break ties by looking at the mask weights
                which_match += 0.5 * mask_weight / tf.reduce_max(mask_weight)
                _, mask_idx = tf.math.top_k(which_match, k=num_to_mask)
            else:
                mask_idx = idx

            mask_idx = tf.sort(mask_idx, 1)
            all_options = tf.stack([
                tf.reshape(input_ids_2d, [-1]),
                tf.fill([self.B * self.L], value=MASK),
                tf.random_uniform(shape=[self.B * self.L], minval=100, maxval=self.vocab_size, dtype=tf.int32),
            ], 1)
            option_to_use = tf.reshape(tf.cast(
                tf.random.categorical(tf.math.log([[0.1, 0.8, 0.1]]), num_samples=self.B * self.L), dtype=tf.float32),
                [self.B * self.L])
            do_mask_option = tf.reshape(tf.reduce_any(tf.equal(mask_idx[..., None],
                                                               sentinel_idx[None, None]), [1]), [-1])
            option_to_use *= tf.cast(do_mask_option, dtype=tf.float32)
            option_to_use = tf.cast(option_to_use, dtype=tf.int32)
            masked_ids = tf.reshape(tf.gather(all_options, option_to_use[:, None], batch_dims=1),
                                    get_shape_list(self.input_ids))

        return {'masked_ids': masked_ids, 'masked_idx': mask_idx}

    def contrastive_loss(self):
        """
        :return:
        """
        contrastive_dim = self.config.get('contrastive_size', self.hidden_size)

        with tf.variable_scope('contrastive'):
            # encode each timestep language-wise
            lang_final_hidden_x = project_and_norm(self.lang_trg_h, hidden_size=contrastive_dim, name='lang_proj',
                                                   add_intermediate=self.config.get('do_projection', False))
            vis_final_hidden_x = project_and_norm(self.img_trg_h, hidden_size=contrastive_dim, name='viz_proj',
                                                  add_intermediate=self.config.get('do_projection', False))

            all_lang_final_hidden_x, my_group_idx = tpu_cross_replica_stack(lang_final_hidden_x)
            num_groups, batch_size, h_ = get_shape_list(all_lang_final_hidden_x, 3)
            tf.logging.info("{} replicas!!! (that we share over)".format(num_groups))
            all_lang_final_hidden_x = tf.reshape(all_lang_final_hidden_x, [num_groups * batch_size, contrastive_dim])

            all_viz_final_hidden_x, my_group_idx = tpu_cross_replica_stack(vis_final_hidden_x)
            all_viz_final_hidden_x = tf.reshape(all_viz_final_hidden_x, [num_groups * batch_size, contrastive_dim])

            temp = self.config.get('contrast_temp', 0.05)
            #####################
            losses = {}
            pairs = [
                {'name': 'lang_to_viz', 'x': lang_final_hidden_x, 'y': all_viz_final_hidden_x},
                {'name': 'viz_to_lang', 'x': vis_final_hidden_x, 'y': all_lang_final_hidden_x},
            ]
            labels = tf.range(batch_size) + my_group_idx * batch_size
            for x in pairs:
                logits = tf.matmul(x['x'], x['y'], transpose_b=True) / temp
                raw_loss = raw_cross_entropy_with_logits(logits, labels)
                losses[x['name']] = tf.reduce_mean(raw_loss)

        losses['loss_all'] = self.config.get('contrast_coef', 1.0) * tf.add_n(list(losses.values())) / len(losses)
        return losses['loss_all'], losses

    def mask_loss(self):
        """
        Language loss
        :return:
        """
        hidden_state_flat = tf.reshape(self.encoder_hidden_states['lang'], [self.B * self.L, self.hidden_size])
        mask_idx_flat = tf.reshape(self.lang_mask_info['masked_idx'] + tf.range(self.B)[:, None] * self.L, [-1])
        hidden_state_pooled = one_hot_gather(hidden_state_flat, mask_idx_flat)
        targets_pooled = tf.gather(tf.reshape(self.input_ids, [-1]), mask_idx_flat)
        logits = self.lm_head(hidden_state_pooled)

        raw_loss = raw_cross_entropy_with_logits(logits, labels=targets_pooled)

        # Just in case any pad tokens pop up, remove them
        is_valid = tf.cast(tf.not_equal(targets_pooled, 0), dtype=raw_loss.dtype)
        denom = tf.reduce_sum(is_valid) + 1e-5

        loss = tf.reduce_sum(is_valid * raw_loss) / denom

        is_right = tf.equal(tf.cast(tf.argmax(logits, -1), dtype=tf.int32), targets_pooled)
        is_right_float = tf.cast(is_right, dtype=tf.float32)
        acc = tf.reduce_sum(is_valid * is_right_float) / denom
        losses = {'loss': loss, 'acc': acc}
        return loss, losses

    def allpairs_temporal_logits(self, xa, xb, scope_name='temporal_paired'):
        """
        We will compute whether, for every pair from xa and xb whether xa[i] < xb[j] in time.

        the items could be the final language hidden states, the final vision hidden states, or xa=lang and xb=vision.

        :param x1: [B, num_chunks_in_group, H]
        :param x2: [B, num_chunks_in_group, H]
        :return: [B, num_chunks_in_group ** 2, 4] -logits, where
                 0 - we think they come from different videos
                 1 - we think xa[i] = xb[j]
                 2 - we think xa[i] < xb[j]
                 3 - we think xa[i] > xb[j]
        """
        batch_size, num_chunks_in_group, hidden_size = get_shape_list(xa, 3)
        assert get_shape_list(xa) == [batch_size, self.num_chunks_in_group, self.hidden_size]
        assert get_shape_list(xb) == [batch_size, self.num_chunks_in_group, self.hidden_size]

        with tf.variable_scope(scope_name):
            # lang first, then viz
            xa_tile = tf.tile(xa[:, :, None], [1, 1, self.num_chunks_in_group, 1])
            xa_tile = tf.reshape(xa_tile, [batch_size, self.num_chunks_in_group ** 2, self.hidden_size])

            xb_tile = tf.tile(xb[:, None], [1, self.num_chunks_in_group, 1, 1])
            xb_tile = tf.reshape(xb_tile, [batch_size, self.num_chunks_in_group ** 2, self.hidden_size])

            h_joint = tf.concat([xa_tile, xb_tile], 2)
            h_joint = tf.reshape(h_joint, [batch_size * (self.num_chunks_in_group ** 2), self.hidden_size * 2])
            # Now do the MLP
            h0 = tf.layers.dense(
                h_joint,
                self.hidden_size,
                kernel_initializer=create_initializer(self.config['initializer_range']),
                name='intermediate',
                activation=gelu,
            )
            h0_ln = layer_norm(h0, 'ln0')
            logits = tf.layers.dense(
                h0_ln,
                4,
                kernel_initializer=create_initializer(self.config['initializer_range']),
                name='logits',
            )
            return logits

    def allpairs_temporal_labels(self, video_src_ids):
        # Set up the labels
        xa_idx = tf.tile(tf.range(self.num_chunks_in_group)[:, None], [1, self.num_chunks_in_group])
        xb_idx = tf.tile(tf.range(self.num_chunks_in_group)[None], [self.num_chunks_in_group, 1])

        # 1 if identical
        is_identical = tf.cast(tf.equal(xa_idx, xb_idx), dtype=tf.int32)
        # 2 if less
        is_less = 2 * tf.cast(tf.less(xa_idx, xb_idx), dtype=tf.int32)
        # 3 if greater
        is_greater = 3 * tf.cast(tf.greater(xa_idx, xb_idx), dtype=tf.int32)

        video_src_ids = tf.reshape(video_src_ids, [self.B, self.num_chunks_in_group])
        is_same_video = tf.equal(video_src_ids[:, None], video_src_ids[:, :, None])

        # 0 if not the same video
        labels = tf.where_v2(
            is_same_video,
            is_identical + is_less + is_greater,
            tf.zeros_like(is_identical),
        )
        labels = tf.reshape(labels, [self.B * (self.num_chunks_in_group ** 2)])
        return labels

    def temporal_loss(self, shuffled_idx_img, video_src_ids):
        """
        :param shuffled_idx_img: [B * self.num_chunks_in_group] with the index (shuffled or not)
        :param video_src_ids: [B, self.num_chunks_in_group] the video ID of each segment
        :return:
        """
        # Pool from language and vision
        # For every pair of (hlang, hviz) we will predict
        # (is same video, is same timestep, hlang < hviz, or hlang > hviz)
        h_lang = tf.reshape(self.encoder_hidden_states['lang'],
                            [self.B, self.num_chunks_in_group, self.lang_chunk_length, self.hidden_size])[:, :, 0]
        h_viz = tf.reshape(self.encoder_hidden_states['viz'],
                           [self.B, self.num_chunks_in_group, self.viz_chunk_length, self.hidden_size])[:, :, 0]
        is_easy_viz = tf.reshape(tf.less(shuffled_idx_img, 64), [self.B, self.num_chunks_in_group])

        # Can add other things here
        labels = self.allpairs_temporal_labels(video_src_ids)
        modality_pairs = [
            {'name': 'lang_viz', 'xa': h_lang, 'xb': h_viz, 'is_easy': is_easy_viz},
            {'name': 'viz_viz', 'xa': h_viz, 'xb': h_viz, 'is_easy': is_easy_viz},
        ]

        loss_info = {}
        for x in modality_pairs:
            logits = self.allpairs_temporal_logits(x['xa'], x['xb'], scope_name=x['name'] + '_temporal')

            # Downweight non-shuffled examples by 99%
            is_easy = tf.logical_and(x['is_easy'][:, :, None], x['is_easy'][:, None])
            label_w = tf.cast(tf.logical_not(is_easy), dtype=tf.float32) * 0.99 + 0.01

            label_w = tf.reshape(label_w, [-1])

            raw_loss = raw_cross_entropy_with_logits(logits, labels) * label_w
            temporal_loss = tf.reduce_mean(raw_loss)

            temporal_is_right = tf.equal(tf.cast(tf.argmax(logits, -1), dtype=tf.int32), labels)
            temporal_is_right_f = tf.cast(temporal_is_right, dtype=tf.float32)
            temporal_acc = tf.reduce_sum(temporal_is_right_f * label_w) / (tf.reduce_sum(label_w) + 1e-5)
            loss_info['{}_loss'.format(x['name'])] = temporal_loss
            loss_info['{}_acc'.format(x['name'])] = temporal_acc

        loss_info['loss'] = loss_info['lang_viz_loss']
        if self.config.get('image_shuffle_prob', 0) > 0:
            loss_info['loss'] += loss_info['viz_viz_loss']

        loss = loss_info['loss'] * self.config.get('temporal_coef', 1.0)
        return loss, loss_info


def model_fn_builder(config: NeatConfig):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        if is_training and config.model.get('transpose_input', False):
            tf.logging.info("Transposing input")
            imgs_to_use = tf.transpose(features['images'], [3, 0, 1, 2])
        elif not is_training:
            imgs_to_use = tf.reshape(features['images'], [-1] + config.model['image_size'] + [3])
        else:
            imgs_to_use = features['images']

        model = MerlotModel(
            config=config.model,
            is_training=True,
            image=imgs_to_use,
            input_ids=features['input_ids'],
            use_tpu=config.device['use_tpu'],
            shuffled_idx_img=features.get('shuffled_idx_img', None),
            mask_input=True,
        )
        lang_loss, lang_losses = model.mask_loss()
        contr_loss, contr_losses = model.contrastive_loss()
        if config.model.get('temporal_coef', 1.0) > 0.0:
            temp_loss, temp_losses = model.temporal_loss(features['shuffled_idx_img'], video_src_ids=features['video_src_ids'],)
        else:
            temp_loss = 0.0
            temp_losses = {}

        losses = {f'lang/{k}': v for k, v in lang_losses.items()}
        losses.update({f'attn/{k}': v for k, v in model.attention_log.items()})
        losses.update({f'contr/{k}': v for k, v in contr_losses.items()})
        losses.update({f'temporal/{k}': v for k, v in temp_losses.items()})

        loss = lang_loss + contr_loss + temp_loss


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
            loss=loss,
            optimizer_config=config.optimizer,
            device_config=config.device,
        )
        train_metrics.update(losses)

        host_call = construct_host_call(scalars_to_log=train_metrics,
                                        model_dir=config.device['output_dir'] if mode != tf.estimator.ModeKeys.EVAL
                                        else os.path.join(config.device['output_dir'], 'eval'),
                                        iterations_per_loop=config.device.get('iterations_per_loop', 1000))

        # This could be useful for debugging, but we can take it out.
        if mode == tf.estimator.ModeKeys.PREDICT:
            bsz = params['batch_size']
            features['self_attn_probs'] = tf.reshape(model.encoder_info['self_attn_probs'],
                                                     [bsz, model.B // bsz] + get_shape_list(
                                                         model.encoder_info['self_attn_probs'])[1:])
            features['self_attn_probs'] = tf.reduce_mean(features['self_attn_probs'], [2])

            features['lang_attn_probs'] = tf.reshape(model.lang_transformer_info['self_attn_probs'],
                                                     [bsz] + get_shape_list(
                                                         model.lang_transformer_info['self_attn_probs'])[1:])
            features['masked_ids'] = model.lang_mask_info['masked_ids']
            with tf.name_scope('debug_for_predict'):
                features.update(losses)

            for k in sorted(features.keys()):
                k_shape_list = get_shape_list(features[k])
                if len(k_shape_list) == 0:
                    print(f"Expanding {k}: {k_shape_list}")
                    features[k] = tf.tile(features[k][None], [bsz])
                elif k_shape_list[0] != bsz:
                    print(f"Reshaping {k}: {k_shape_list}")
                    features[k] = tf.reshape(features[k], [bsz, -1])

                if features[k].dtype == tf.bfloat16:
                    features[k] = tf.cast(features[k], dtype=tf.float32)

            return tf.contrib.tpu.TPUEstimatorSpec(mode=tf.estimator.ModeKeys.PREDICT,
                                                   predictions=features)
        elif mode == tf.estimator.ModeKeys.EVAL:
            keys_sorted = sorted(losses.keys())
            values_sorted = [tf.tile(losses[k][None], [model.batch_size]) for k in keys_sorted]

            def metric_fn(*args):
                return {k: tf.metrics.mean(args[i], name=f'{k}_avg') for i, k in enumerate(keys_sorted)}

            return tf.contrib.tpu.TPUEstimatorSpec(mode=tf.estimator.ModeKeys.EVAL,
                                                   loss=loss,
                                                   train_op=train_op,
                                                   eval_metrics=(metric_fn, values_sorted),
                                                   scaffold_fn=scaffold_fn,
                                                   host_call=host_call)

        return tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            eval_metrics=None,
            scaffold_fn=scaffold_fn,
            host_call=host_call)

    return model_fn
