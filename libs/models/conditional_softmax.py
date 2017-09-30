#! /usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from .deliberation import DelibNMT, DelibInitializer
from ..layers import *

__author__ = 'fyabc'


class ConditionalSoftmaxInitializer(DelibInitializer):
    def init_params(self):
        np_parameters = super(ConditionalSoftmaxInitializer, self).init_params()

        self.init_decoder(np_parameters)    # todo: need test

        return np_parameters


class ConditionalSoftmaxModel(DelibNMT):
    def __init__(self, options, given_params=None):
        super(ConditionalSoftmaxModel, self).__init__(options, given_params)

        # Share context info between RNN and per-word prediction.
        self._ctx_info = None

    def get_context_info(self, context, x_mask, trg_feature):
        ctx_info = super(ConditionalSoftmaxModel, self).get_context_info(context, x_mask, trg_feature)
        self._ctx_info = ctx_info
        return ctx_info

    def trainable_parameters(self):
        # todo: remove per-word prediction parameters from trainable parameters
        return

    def build_model(self, set_instance_variables=False):
        """Build a training model."""

        dropout_rate = self.O['use_dropout']

        opt_ret = {}

        trng = RandomStreams(1234)
        use_noise = theano.shared(np.float32(1.))

        if dropout_rate is not False:
            dropout_params = [use_noise, trng, dropout_rate]
        else:
            dropout_params = None

        # Encoder.
        (x, x_mask, y, y_mask), context, _ = self.input_to_context(dropout_params=dropout_params)

        # Per-word prediction decoder.
        # todo: change y_pos_ into expression of y?
        y_pos_ = T.matrix('y_pos_', dtype='int64')
        tgt_pos_embed = self.P['Wemb_dec_pos'][y_pos_.flatten()].reshape(
            [y_pos_.shape[0], y_pos_.shape[1], self.O['dim_word']])
        pw_probs = self.independent_decoder(tgt_pos_embed, y, y_mask, context, x_mask,
                                            dropout_params=None, trng=trng, use_noise=use_noise)

        # RNN decoder.
        n_timestep, n_timestep_tgt, n_samples = self.input_dimensions(x, y)

        # Initial decoder state
        init_decoder_state = self.feed_forward(self._ctx_info, prefix='ff_state', activation=tanh)

        # Word embedding (target), we will shift the target sequence one time step
        # to the right. This is done because of the bi-gram connections in the
        # readout and decoder rnn. The first target will be all zeros and we will
        # not condition on the last output.
        tgt_embedding = self.embedding(y, n_timestep_tgt, n_samples, 'Wemb_dec')
        emb_shifted = T.zeros_like(tgt_embedding)
        emb_shifted = T.set_subtensor(emb_shifted[1:], tgt_embedding[:-1])
        tgt_embedding = emb_shifted
        pre_projected_context = self.attention_projected_context(context, prefix='decoder')

        # Decoder - pass through the decoder conditional gru with attention
        hidden_decoder, context_decoder, opt_ret['dec_alphas'], _ = self.decoder(
            tgt_embedding, y_mask=y_mask, init_state=init_decoder_state, context=context, x_mask=x_mask,
            projected_context=pre_projected_context, dropout_params=dropout_params, one_step=False,
        )

        # todo: change this into conditional softmax.
        trng, use_noise, probs = self.get_word_probability(hidden_decoder, context_decoder, tgt_embedding,
                                                           trng=trng, use_noise=use_noise)

        # [NOTE]: Only use RNN decoder loss.
        test_cost = self.build_cost(y, y_mask, probs)
        cost = test_cost / self.O['cost_normalization']  # cost used to derive gradient in training

        return trng, use_noise, x, x_mask, y, y_mask, y_pos_, opt_ret, cost, test_cost, probs
