#! /usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from .deliberation import DelibNMT, DelibInitializer
from ..utility.theano_arg_top_k import theano_argpartsort
from ..utility.utils import delib_env
from ..layers.basic import tanh


class ConditionalSoftmaxInitializer2(DelibInitializer):
    def __init__(self, options, delib_options):
        super(ConditionalSoftmaxInitializer2, self).__init__(options)

        # Per-word prediction options
        self.DO = delib_options

    def init_params(self):
        np_parameters = super(ConditionalSoftmaxInitializer2, self).init_params()

        # todo

        return np_parameters


class ConditionalSoftmaxModel2(DelibNMT):
    def __init__(self, options, delib_options, given_params=None):
        super(ConditionalSoftmaxModel2, self).__init__(options, given_params=given_params)
        self.initializer = ConditionalSoftmaxInitializer2(options, delib_options)

        self.DO = delib_options

        self._check_options()

    def _check_options(self):
        assert self.O['dim_word'] == self.DO['dim_word'], 'Word dimensions must be same between two decoders'

    def build_model(self, set_instance_variables=False):
        """
        Build a training model.

        Parameters
        ----------
        set_instance_variables

        Returns
        -------

        """

        dropout_rate = self.O['use_dropout']

        opt_ret = {}

        trng = RandomStreams(1234)
        use_noise = theano.shared(np.float32(1.))

        if dropout_rate is not False:
            dropout_params = [use_noise, trng, dropout_rate]
        else:
            dropout_params = None

        # Encoder.
        # [NOTE] Encoder options of self.O and self.DO must be same. Switch to use self.DO['use_src_pos'].
        with delib_env(self, 'use_src_pos'):
            (x, x_mask, y, y_mask), context, _ = self.input_to_context(dropout_params=dropout_params)

        n_timestep, n_timestep_tgt, n_samples = self.input_dimensions(x, y)

        # Word embedding (target).
        tgt_embedding = self.embedding(y, n_timestep_tgt, n_samples, 'Wemb_dec')

        # Target embedding (with position embedding).
        y_pos_ = T.repeat(T.arange(y.shape[0]).dimshuffle(0, 'x'), y.shape[1], 1)
        tgt_pos_embed = self.P['Wemb_dec_pos'][y_pos_.flatten()].reshape(
            [y_pos_.shape[0], y_pos_.shape[1], self.DO['dim_word']])
        tgt_embedding += tgt_pos_embed

        # We will shift the target sequence one time step
        # to the right. This is done because of the bi-gram connections in the
        # readout and decoder rnn. The first target will be all zeros and we will
        # not condition on the last output.
        emb_shifted = T.zeros_like(tgt_embedding)
        emb_shifted = T.set_subtensor(emb_shifted[1:], tgt_embedding[:-1])
        tgt_embedding = emb_shifted
        pre_projected_context = self.attention_projected_context(context, prefix='decoder')

        # Context info.
        context_info = self.get_context_info(context, x_mask, tgt_embedding)
        if self.O['use_attn']:
            # Use target attention, 3-d context info; else context mean, 2-d
            context_mean = T.mean(context_info, axis=0)
        else:
            context_mean = context_info
        init_decoder_state = self.feed_forward(context_mean, prefix='ff_state', activation=tanh)

        # Per-word prediction decoder.
        with delib_env(self):
            pw_probs = self.independent_decoder(tgt_embedding, y_mask, context, x_mask,
                                                dropout_params=None, trng=trng, use_noise=use_noise)

        # RNN Decoder - pass through the decoder conditional gru with attention
        hidden_decoder, context_decoder, opt_ret['dec_alphas'], _ = self.decoder(
            tgt_embedding, y_mask=y_mask, init_state=init_decoder_state, context=context, x_mask=x_mask,
            projected_context=pre_projected_context, dropout_params=dropout_params, one_step=False,
        )

        pass

    def independent_decoder(self, tgt_embedding, y_mask, context, x_mask, dropout_params=None, **kwargs):
        """
        Build per-word prediction decoder in conditional softmax model.

        Parameters
        ----------
        tgt_embedding : Theano variable
            ([Tt], [Bs])
        y_mask : Theano variable
            ([Tt], [Bs])
        context
        x_mask
        dropout_params
        kwargs

        Returns
        -------

        Notes
        -----
        This method must be called in context `_delib_env`.
        """

        # todo

        if self.O['decoder_all_attention']:
            projected_context = T.dot(context, self.P['attn_0_ctx2hidden'])
        if self.O['decoder_style'] == 'stackNN':
            ctx_info = self.get_context_info(context, x_mask, tgt_pos_embed)
            H_ = T.dot(tgt_pos_embed, self.P['decoder_W_pose2h']) + \
                T.dot(ctx_info, self.P['decoder_W_ctx2h']) + self.P['decoder_b_i2h']
            for layer_id in xrange(self.O['n_decoder_layers']):
                H_ = T.tanh(H_)
                if self.O['decoder_all_attention']:
                    ctx_info = self.attention_layer(context, x_mask, projected_context, H_, layer_id)
                    H_ = T.dot(H_, self.P['decoder_W_h2h'][layer_id]) + \
                        T.dot(ctx_info, self.P['decoder_W_att2h'][layer_id]) + \
                        self.P['decoder_b_h2h'][layer_id]
                else:
                    H_ = T.dot(H_, self.P['decoder_W_h2h'][layer_id]) + self.P['decoder_b_h2h'][layer_id]
            H_ = T.tanh(H_)
        elif self.O['decoder_style'] == 'stackLSTM':
            H_ = self.stackLSTM(tgt_pos_embed, 'decoder', y_mask, context, x_mask)
        else:
            raise Exception('Not implemented yet')
        trng = kwargs.pop('trng', RandomStreams(1234))
        use_noise = kwargs.pop('use_noise', theano.shared(np.float32(1.)))
        if self.O['use_dropout']:
            H_ = self.dropout(H_, use_noise, trng)

        logit = self.feed_forward(H_, prefix='fc_lastHtoSoftmax', activation=lambda x: x)
        logit_shp = logit.shape

        unnormalized_probs = logit.reshape([-1, logit_shp[2]])
        return unnormalized_probs
