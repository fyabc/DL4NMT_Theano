# -*- coding: utf-8 -*-

from __future__ import print_function

from collections import OrderedDict

import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from ..layers.basic import _slice
from .model import ParameterInitializer, NMTModel
from ..utility.utils import *
from ..layers import *
from ..constants import fX, profile


class DelibInitializer(ParameterInitializer):
    def init_independent_decoder(self, np_parameters):
        import math
        dim_word = self.O['dim_word']
        dim = self.O['dim']
        n_dec_layers = self.O['n_decoder_layers']

        if self.O['decoder_style'] == 'stackNN':
            np_parameters['decoder_W_pose2h'] = normal_weight(dim_word, dim_word)
            np_parameters['decoder_W_ctx2h'] = normal_weight(2 * dim, dim_word, scale=1. / math.sqrt(2 * dim))
            np_parameters['decoder_b_i2h'] = np.zeros((dim_word,), dtype=np.float32)

            np_parameters['decoder_W_h2h'] = 1. / math.sqrt(dim_word) * \
                                             np.random.rand(n_dec_layers, dim_word, dim_word).astype('float32')
            np_parameters['decoder_b_h2h'] = np.zeros((n_dec_layers, dim_word)).astype('float32')
        elif self.O['decoder_style'] == 'stackLSTM':
            for layer_id in xrange(n_dec_layers):
                np_parameters[_p('decoder', 'W', 'lstm_i2h', layer_id)] = np.zeros((3 * dim + dim_word, 4 * dim),
                                                                                   dtype='float32')
                np_parameters[_p('decoder', 'b', 'lstm_i2h', layer_id)] = np.zeros((4 * dim,), dtype='float32')
                for ii_ in xrange(3):
                    for jj_ in xrange(4):
                        np_parameters[_p('decoder', 'W', 'lstm_i2h', layer_id)][
                            ii_ * dim:(ii_ + 1) * dim, jj_ * dim:(jj_ + 1) * dim] = normal_weight(dim)
                np_parameters[_p('decoder', 'W', 'lstm_i2h', layer_id)][3 * dim:, :] = \
                    np.random.rand(dim_word, 4 * dim) / math.sqrt(dim_word)
            np_parameters[_p('decoder', 'W', 'TolastH')] = np.concatenate([
                np.random.rand(dim, dim_word).astype('float32') / math.sqrt(dim),
                np.random.rand(dim, dim_word).astype('float32') / math.sqrt(dim),
                np.random.rand(dim, dim_word).astype('float32') / math.sqrt(dim),
                np.random.rand(dim_word, dim_word).astype('float32') / math.sqrt(dim_word),
            ], axis=0)
            np_parameters[_p('decoder', 'b', 'TolastH')] = np.zeros((dim_word,), dtype='float32')

        else:
            raise Exception('Not implemented yet')
        self.init_feed_forward(np_parameters, 'fc_lastHtoSoftmax', dim_word, self.O['n_words'], False)

        if self.O['decoder_all_attention'] or self.O['use_attn']:
            np_parameters['attn_0_ctx2hidden'] = normal_weight(2 * dim, dim, scale=1. / math.sqrt(2 * dim))

        if self.O['use_attn']:
            np_parameters['attn_0_pose2hidden'] = normal_weight(dim_word, dim, scale=0.01)
            np_parameters['attn_0_b'] = np.zeros((dim,), dtype='float32')
            np_parameters['attn_1_W'] = np.random.rand(dim).astype('float32') * 1. / math.sqrt(dim)
            np_parameters['attn_1_b'] = np.zeros((1,), dtype='float32')

        if self.O['decoder_all_attention']:
            np_parameters['decoder_attn_0_h2h'] = np.array([normal_weight(dim_word, dim, scale=0.01)
                                                            for _ in xrange(n_dec_layers)], dtype=fX)
            np_parameters['decoder_attn_0_b'] = np.zeros((n_dec_layers, dim), dtype=fX)
            np_parameters['decoder_attn_1_W'] = np.random.rand(n_dec_layers, dim).astype(fX) * 1. / math.sqrt(dim)
            np_parameters['decoder_attn_1_b'] = np.zeros((n_dec_layers, 1), dtype=fX)
            np_parameters['decoder_W_att2h'] = np.array([normal_weight(2 * dim, dim_word, scale=1. / math.sqrt(2 * dim))
                                                         for _ in xrange(n_dec_layers)], dtype=fX)

    def init_params(self):
        np_parameters = OrderedDict()

        # Source embedding
        self.init_embedding(np_parameters, 'Wemb', self.O['n_words_src'], self.O['dim_word'])

        # Source position embedding
        if self.O['use_src_pos']:
            self.init_embedding(np_parameters, 'Wemb_pos', self.O['maxlen'] + 1, self.O['dim_word'])

        # Encoder: bidirectional RNN
        np_parameters = self.init_encoder(np_parameters)

        # Target position embedding
        self.init_embedding(np_parameters, 'Wemb_dec_pos', self.O['maxlen'] + 1, self.O['dim_word'])

        # Decoder
        self.init_independent_decoder(np_parameters)

        return np_parameters


class DelibNMT(NMTModel):
    def __init__(self, options, given_params=None):
        super(DelibNMT, self).__init__(options, given_params)
        self.initializer = DelibInitializer(options)

    def get_context_info(self, context, x_mask, trg_feature):
        """
        Get context info.

        Parameters
        ----------
        context         ([Ts], [Bs], [Hc])
        x_mask          ([Ts], [Bs])
        trg_feature     ([Tt], [Bs], [W])

        Returns
        -------
        ctx_info: Theano tensor variable
        if use_attn:    ([Tt], [Bs], [Hc])
        else:           ([Bs], [Hc])
        """

        if self.O['use_attn']:
            tmp = T.tanh(T.dot(context, self.P['attn_0_ctx2hidden']) +
                         T.dot(trg_feature, self.P['attn_0_pose2hidden']).dimshuffle(0, 'x', 1, 2) +
                         self.P['attn_0_b'])
            tmp = T.dot(tmp, self.P['attn_1_W']).dimshuffle(0, 1, 2, 'x') + self.P['attn_1_b']
            tmp = T.exp(tmp)
            tmp = tmp.reshape([tmp.shape[0], tmp.shape[1], tmp.shape[2]])
            tmp *= x_mask.dimshuffle('x', 0, 1)
            weight = tmp / tmp.sum(axis=1, keepdims=True)
            ctx_info = (weight.dimshuffle(0, 1, 2, 'x') * context).sum(axis=1)
        else:
            ctx_info = self.get_context_mean(context, x_mask)
        return ctx_info

    def attention_layer(self, context, x_mask, projected_context, h, layer_id):
        tmp = T.tanh(projected_context +
                     T.dot(h, self.P['decoder_attn_0_h2h'][layer_id]).dimshuffle(0, 'x', 1, 2) +
                     self.P['decoder_attn_0_b'][layer_id])
        tmp = T.dot(tmp, self.P['decoder_attn_1_W'][layer_id]).dimshuffle(0, 1, 2, 'x') + \
            self.P['decoder_attn_1_b'][layer_id]
        tmp = T.exp(tmp)
        tmp = tmp.reshape([tmp.shape[0], tmp.shape[1], tmp.shape[2]])
        tmp *= x_mask.dimshuffle('x', 0, 1)
        weight = tmp / tmp.sum(axis=1, keepdims=True)
        ctx_info = (weight.dimshuffle(0, 1, 2, 'x') * context).sum(axis=1)

        return ctx_info

    def _lstm_step(self, mask_, input_, h_, c_):
        """
        Since this function is not used in ``scan'' mode,  we do not need to much extra vars
        That is, input_ could be W*[h_, x_] or W*[h_, x_, ctx_]
        Following PaddlePaddle, they are pre-computed
        """

        dim = self.O['dim']
        i1 = T.nnet.sigmoid(_slice(input_, 0, dim))
        f1 = T.nnet.sigmoid(_slice(input_, 1, dim))
        o1 = T.nnet.sigmoid(_slice(input_, 2, dim))
        c1 = T.tanh(_slice(input_, 3, dim))
        c1 = f1 * c_ + i1 * c1
        c1 = mask_[:, :, None] * c1 + (1. - mask_)[:, :, None] * c_
        h1 = o1 * T.tanh(c1)
        h1 = mask_[:, :, None] * h1 + (1. - mask_)[:, :, None] * h_
        return h1, c1

    def stackLSTM(self, state_below, prefix='lstm', mask=None, context=None, context_mask=None, **kwargs):
        """
        Please note that in this setting, any layer before applying ctx_vec are non-linear transformations of position
        embedding. As as result, I think it is not of much practical value to ``multi-lstm'' the layers before ctx. The
        two possible structures are:
            (i) lstm + single attention
            (ii) lstm + multiple, even all attention
        I just implement (i) here
        """

        dim = self.O['dim']
        assert context, 'Context must be provided'
        assert context.ndim == 3, 'Context must be 3-d: #annotation * #sample * dim'
        n_steps = state_below.shape[0]
        n_samples = state_below.shape[1]
        ctx_info = self.get_context_info(context, context_mask, state_below)
        H_ = T.zeros([n_steps, n_samples, dim], dtype=theano.config.floatX)
        C_ = T.zeros([n_steps, n_samples, dim], dtype=theano.config.floatX)
        for layer_id in xrange(self.O['n_decoder_layers']):
            # DO NOT change the order of the concated matrices !!!!
            # todo: all attention: add get_context_info into each layer
            input_ = T.dot(
                concatenate([H_, ctx_info, state_below], axis=2),
                self.P[_p(prefix, 'W', 'lstm_i2h', layer_id)]
            ) + self.P[_p(prefix, 'b', 'lstm_i2h', layer_id)]
            H_, C_ = self._lstm_step(mask, input_, H_, C_)
        H_ = T.dot(
            concatenate([H_, ctx_info, state_below], axis=2),
            self.P[_p(prefix, 'W', 'TolastH')]) + self.P[_p(prefix, 'b', 'TolastH')]
        return H_

    def independent_decoder(self, tgt_pos_embed, y, y_mask, context, x_mask, **kwargs):
        """

        Parameters
        ----------
        tgt_pos_embed
        y
        y_mask:         ([Tt], [Bs])
        context:        ([Ts], [Bs], [Hc])
        x_mask:         ([Ts], [Bs])
        kwargs

        Returns
        -------

        """

        if self.O['delib_reversed'] in ('decoder', 'all'):
            context = context[::-1]
            x_mask = x_mask[::-1]

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
        if kwargs.pop('softmax', True):
            probs = T.nnet.softmax(logit.reshape([-1, logit_shp[2]]))
            return probs
        else:
            unnormalized_probs = logit.reshape([-1, logit_shp[2]])
            return unnormalized_probs

    def build_cost(self, y, y_mask, probs, epsilon=0.0):
        """Build the cost from probabilities and target."""

        from ..utility.basic import floatX
        epsilon = floatX(epsilon)

        y_flat = y.flatten()
        y_flat_idx = T.arange(y_flat.shape[0]) * self.O['n_words'] + y_flat
        cost = -T.log(probs.flatten()[y_flat_idx] + epsilon)
        cost = cost.reshape([y.shape[0], y.shape[1]])
        cost = (cost * y_mask).sum() / y_mask.sum()
        return cost

    def build_model(self, set_instance_variables=False):
        """Build a training model."""

        # todo: add reversed decoder

        dropout_rate = self.O['use_dropout']

        opt_ret = {}

        trng = RandomStreams(1234)
        use_noise = theano.shared(np.float32(1.))

        if dropout_rate is not False:
            dropout_params = [use_noise, trng, dropout_rate]
        else:
            dropout_params = None

        (x, x_mask, y, y_mask), context, _ = self.input_to_context(dropout_params=dropout_params)
        y_pos_ = T.matrix('y_pos_', dtype='int64')
        tgt_pos_embed = self.P['Wemb_dec_pos'][y_pos_.flatten()].reshape(
            [y_pos_.shape[0], y_pos_.shape[1], self.O['dim_word']])
        probs = self.independent_decoder(tgt_pos_embed, y, y_mask, context, x_mask,
                                         trng=trng, use_noise=use_noise)

        # [NOTE]: difference between ``DelibNMT.build_cost`` and ``NMTModel.build_cost``
        test_cost = self.build_cost(y, y_mask, probs)
        cost = test_cost / self.O['cost_normalization']  # cost used to derive gradient in training

        return trng, use_noise, x, x_mask, y, y_mask, y_pos_, opt_ret, cost, test_cost, probs

    def build_sampler(self, **kwargs):
        # Build model
        _, use_noise, x, x_mask, y, y_mask, y_pos_, _, cost, _, probs = self.build_model()
        n_timestep, n_timestep_tgt, n_samples = self.input_dimensions(x, y)

        use_noise.set_value(0.)
        inps = [x, x_mask]

        if self.O['decoder_style'] == 'stackNN':
            givens = [(y_pos_, T.repeat(T.arange(n_timestep).dimshuffle(0, 'x'), n_samples, 1))]
        elif self.O['decoder_style'] == 'stackLSTM':
            givens = [
                (y_pos_, T.repeat(T.arange(n_timestep).dimshuffle(0, 'x'), n_samples, 1)),
                (y_mask, T.fill(x_mask, 1.0).astype(fX))
            ]
        else:
            raise NotImplementedError('This decoder style not implemented yet')
        print('Build predictor... ', end='')
        f_predictor = theano.function(
            inps, probs,
            givens=givens,
            profile=profile
        )
        print('Done')
        return f_predictor, None

    def gen_batch_sample(self, f_init, f_next, x, x_mask, trng=None, k=1, maxlen=30, eos_id=0, attn_src=False, **kwargs):
        # 解决包含y_mask和y_pos的问题
        # 一个方案：translate的时候，翻译到第几个词，就给出第几个词的pos，翻译到出了EOS的时候就结束。

        if attn_src is True:
            raise NotImplementedError('attn_src not implemented in DelibNMT yet')

        f_predictor = f_init

        # FIXME: truncate source sentences to fix the bug: source embedding out of bound.
        x, x_mask = x[:self.O['maxlen'] + 1], x_mask[:self.O['maxlen'] + 1]

        probs = f_predictor(x, x_mask)
        predict = probs.argmax(axis=1).reshape((x.shape[0], x.shape[1]))

        return predict.T.tolist()


__all__ = [
    'DelibInitializer',
    'DelibNMT',
]
