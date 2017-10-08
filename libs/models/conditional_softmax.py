#! /usr/bin/python
# -*- coding: utf-8 -*-

from collections import OrderedDict
from contextlib import contextmanager

import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from .deliberation import DelibNMT, DelibInitializer
from ..layers import *
from ..utility.basic import floatX
from ..utility.utils import _p

__author__ = 'fyabc'


@contextmanager
def _delib_env(obj):
    """Context manager to set the per-word prediction options `obj.DO` as options."""

    tmp_o = obj.O
    obj.O = obj.DO
    try:
        yield
    finally:
        obj.DO = obj.O
        obj.O = tmp_o


class ConditionalSoftmaxInitializer(DelibInitializer):
    def __init__(self, options, delib_options):
        super(ConditionalSoftmaxInitializer, self).__init__(options)

        # Per-word prediction options
        self.DO = delib_options

    def init_params(self):
        np_parameters = OrderedDict()

        # RNN part.

        # Source embedding
        self.init_embedding(np_parameters, 'Wemb', self.O['n_words_src'], self.O['dim_word'])

        # Encoder: bidirectional RNN
        np_parameters = self.init_encoder(np_parameters)

        # Target embedding
        self.init_embedding(np_parameters, 'Wemb_dec', self.O['n_words'], self.O['dim_word'])

        # Decoder
        self.init_decoder(np_parameters)

        # Readout
        context_dim = 2 * self.O['dim']
        np_parameters = self.init_feed_forward(np_parameters, prefix='ff_logit_lstm', nin=self.O['dim'],
                                               nout=self.O['dim_word'], orthogonal=False)
        np_parameters = self.init_feed_forward(np_parameters, prefix='ff_logit_prev', nin=self.O['dim_word'],
                                               nout=self.O['dim_word'], orthogonal=False)
        np_parameters = self.init_feed_forward(np_parameters, prefix='ff_logit_ctx', nin=context_dim,
                                               nout=self.O['dim_word'], orthogonal=False)
        np_parameters = self.init_feed_forward(np_parameters, prefix='ff_logit', nin=self.O['dim_word'],
                                               nout=self.O['n_words'])

        # Per-word prediction part.

        with _delib_env(self):
            # Source position embedding
            if self.O['use_src_pos']:
                self.init_embedding(np_parameters, 'Wemb_pos', self.O['maxlen'] + 1, self.O['dim_word'])

            # Target position embedding
            self.init_embedding(np_parameters, 'Wemb_dec_pos', self.O['maxlen'] + 1, self.O['dim_word'])

            # Deliberation decoder
            self.init_independent_decoder(np_parameters)

        return np_parameters


class ConditionalSoftmaxModel(DelibNMT):
    def __init__(self, options, delib_options, given_params=None):
        super(ConditionalSoftmaxModel, self).__init__(options, given_params)
        self.initializer = ConditionalSoftmaxInitializer(options, delib_options)

        # Per-word prediction options
        self.DO = delib_options

        # Share context info between RNN and per-word prediction.
        self._ctx_info = None

    def get_context_info(self, context, x_mask, trg_feature):
        ctx_info = super(ConditionalSoftmaxModel, self).get_context_info(context, x_mask, trg_feature)
        self._ctx_info = ctx_info
        return ctx_info

    def trainable_parameters(self):
        # todo: remove per-word prediction parameters from trainable parameters
        params = super(ConditionalSoftmaxModel, self).trainable_parameters()

        to_be_deleted = set()

        if self.DO['decoder_style'] == 'stackNN':
            to_be_deleted |= {
                'decoder_W_pose2h', 'decoder_W_ctx2h',
                'decoder_b_i2h', 'decoder_W_h2h',
                'decoder_b_h2h',
            }
        elif self.DO['decoder_style'] == 'stackLSTM':
            for layer_id in xrange(self.DO['n_decoder_layers']):
                to_be_deleted |= {_p('decoder', 'W', 'lstm_i2h', layer_id), _p('decoder', 'b', 'lstm_i2h', layer_id)}
            to_be_deleted |= {_p('decoder', 'W', 'TolastH'), _p('decoder', 'b', 'TolastH')}
        else:
            raise Exception('Not implemented yet')

        to_be_deleted |= {'fc_lastHtoSoftmax_W', 'fc_lastHtoSoftmax_b'}
        if self.DO['decoder_all_attention'] or self.DO['use_attn']:
            to_be_deleted.add('attn_0_ctx2hidden')
        if self.DO['use_attn']:
            to_be_deleted |= {'attn_0_pose2hidden', 'attn_0_b', 'attn_1_W', 'attn_1_b'}
        if self.DO['decoder_all_attention']:
            to_be_deleted |= {'decoder_attn_0_h2h', 'decoder_attn_0_b', 'decoder_attn_1_W',
                              'decoder_attn_1_b', 'decoder_W_att2h'}

        for k in to_be_deleted:
            del params[k]
        return params

    def independent_decoder(self, tgt_pos_embed, y, y_mask, context, x_mask, dropout_params=None, **kwargs):
        """
        Build per-word prediction decoder.

        Parameters
        ----------
        tgt_pos_embed
        y : Theano variable
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
        with _delib_env(self):
            # todo: change y_pos_ into expression of y?
            y_pos_ = T.matrix('y_pos_', dtype='int64')
            tgt_pos_embed = self.P['Wemb_dec_pos'][y_pos_.flatten()].reshape(
                [y_pos_.shape[0], y_pos_.shape[1], self.DO['dim_word']])
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

        trng, use_noise, probs = self.get_word_probability(hidden_decoder, context_decoder, tgt_embedding,
                                                           trng=trng, use_noise=use_noise, pw_probs=pw_probs)

        # [NOTE]: Only use RNN decoder loss.
        test_cost = self.build_cost(y, y_mask, probs)
        cost = test_cost / self.O['cost_normalization']  # cost used to derive gradient in training

        return trng, use_noise, x, x_mask, y, y_mask, y_pos_, opt_ret, cost, test_cost, probs

    def get_word_probability(self, hidden_decoder, context_decoder, tgt_embedding, **kwargs):
        """Compute word probabilities.

        Parameters
        ----------
        hidden_decoder
        context_decoder
        tgt_embedding
        kwargs

        Returns
        -------
            probs: numpy array, ([Tt] * [Bs], n_words)
        """

        trng = kwargs.pop('trng', RandomStreams(1234))
        use_noise = kwargs.pop('use_noise', theano.shared(np.float32(0.)))

        logit_lstm = self.feed_forward(hidden_decoder, prefix='ff_logit_lstm', activation=linear)
        logit_prev = self.feed_forward(tgt_embedding, prefix='ff_logit_prev', activation=linear)
        logit_ctx = self.feed_forward(context_decoder, prefix='ff_logit_ctx', activation=linear)
        logit = T.tanh(logit_lstm + logit_prev + logit_ctx)  # n_timestep * n_sample * dim_word
        if self.O['dropout_out']:
            logit = self.dropout(logit, use_noise, trng, self.O['dropout_out'])

        pw_probs = kwargs.pop('pw_probs', None)     # ([Tt] * [Bs], n_words)
        k = self.O['cond_softmax_k']

        # n_timestep * n_sample * n_words
        logit = self.feed_forward(logit, prefix='ff_logit', activation=linear)
        logit_shp = logit.shape
        logit_reshaped = logit.reshape([logit_shp[0] * logit_shp[1], logit_shp[2]])

        if pw_probs is not None:
            top_k_args = T.argsort(-pw_probs)[:, :k]
            # [NOTE] indices to get/set top-k value
            top_k_indices = (T.arange(pw_probs.shape[0]).dimshuffle([0, 'x']), top_k_args)

            probs = T.alloc(floatX(0.), *pw_probs.shape)
            # get top-k probability indices from per-word probs, then apply it into probs
            T.set_subtensor(probs[top_k_indices], T.nnet.softmax(logit_reshaped[top_k_indices]))
        else:
            probs = T.nnet.softmax(logit_reshaped)

        return trng, use_noise, probs
