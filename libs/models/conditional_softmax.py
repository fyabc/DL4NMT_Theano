#! /usr/bin/python
# -*- coding: utf-8 -*-

from collections import OrderedDict

import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from .deliberation import DelibNMT, DelibInitializer, NMTModel
from ..layers import *
from ..layers.basic import _slice
from ..constants import fX, profile
from ..utility.basic import floatX
from ..utility.utils import *
from ..utility.theano_ops import theano_argpartsort, theano_unique

__author__ = 'fyabc'


class ConditionalSoftmaxInitializer(DelibInitializer):
    def __init__(self, options):
        super(ConditionalSoftmaxInitializer, self).__init__(options)

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

        # Source position embedding
        if self.O['use_src_pos']:
            self.init_embedding(np_parameters, 'Wemb_pos', self.O['maxlen'] + 1, self.O['dim_word'])

        # Target position embedding
        self.init_embedding(np_parameters, 'Wemb_dec_pos', self.O['maxlen'] + 1, self.O['dim_word'])

        # Deliberation decoder
        self.init_independent_decoder(np_parameters)

        return np_parameters

    def init_independent_decoder(self, np_parameters):
        import math
        dim_word = self.O['dim_word']
        dim = self.O['dim']
        n_dec_layers = self.O['n_decoder_layers']

        if self.O['decoder_style'] == 'stackNN':
            np_parameters['decoder_W_pose2h'] = normal_weight(dim_word, dim_word)
            np_parameters['decoder_W_ctx2h'] = normal_weight(2 * dim, dim_word, scale=1. / math.sqrt(2 * dim))
            np_parameters['decoder_b_i2h'] = np.zeros((dim_word,), dtype=np.float32)

            # np_parameters['decoder_W_h2h'] = 1. / math.sqrt(dim_word) * \
            #                                  np.random.rand(n_dec_layers, dim_word, dim_word).astype('float32')
            # np_parameters['decoder_b_h2h'] = np.zeros((n_dec_layers, dim_word)).astype('float32')
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
        # self.init_feed_forward(np_parameters, 'fc_lastHtoSoftmax', dim_word, self.O['n_words'], False)

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
            # [NOTE] can share it with <Wc>?
            np_parameters['decoder_W_att2h'] = np.array([normal_weight(2 * dim, dim, scale=1. / math.sqrt(2 * dim))
                                                         for _ in xrange(n_dec_layers)], dtype=fX)


class ConditionalSoftmaxModel(DelibNMT):
    def __init__(self, options, delib_options, given_params=None):
        super(ConditionalSoftmaxModel, self).__init__(options, given_params)
        self.initializer = ConditionalSoftmaxInitializer(options)

        self._check_options(delib_options)

    def _check_options(self, delib_options):
        if delib_options is None:
            return

        def _assert_same(key):
            assert self.O[key] == delib_options[key], \
                'Option "{}" not same ({} vs {})'.format(key, self.O[key], delib_options[key])

        def _move(key):
            if self.O[key] != delib_options[key]:
                message('WARNING: move option "{}" value {} from deliberation'.format(key, delib_options[key]))
                self.O[key] = delib_options[key]

        _assert_same('dim')
        _assert_same('dim_word')
        _assert_same('maxlen')
        _assert_same('n_encoder_layers')
        _assert_same('n_decoder_layers')
        _assert_same('decoder_all_attention')
        _move('use_src_pos')
        _move('decoder_style')
        _move('use_attn')
        _move('delib_reversed')

    def build_model(self, set_instance_variables=False):
        """
        Build a training model.

        Parameters
        ----------
        set_instance_variables

        Returns
        -------

        """

        opt_ret, trng, use_noise, dropout_params = self._prepare_encoder()

        # Encoder.
        # [NOTE] Encoder options of self.O and self.DO must be same. Switch to use self.DO['use_src_pos'].
        (x, x_mask, y, y_mask), context, _ = self.input_to_context(dropout_params=dropout_params)

        n_timestep, n_timestep_tgt, n_samples = self.input_dimensions(x, y)

        tgt_embedding = self.embedding(y, n_timestep_tgt, n_samples, 'Wemb_dec')
        tgt_embedding += self._pos_embedding(n_timestep_tgt, n_samples, 'Wemb_dec_pos')
        tgt_embedding = self._shift_embedding(tgt_embedding)

        pre_projected_context = self.attention_projected_context(context, prefix='decoder')

        # Context info.
        context_info = self.get_context_info(context, x_mask, tgt_embedding)
        if self.O['use_attn']:
            # Use target attention, 3-d context info; else context mean, 2-d
            # [NOTE] need test
            context_mean = T.mean(context_info, axis=0)
        else:
            context_mean = context_info
        init_decoder_state = self.feed_forward(context_mean, prefix='ff_state')

        # Per-word prediction decoder.
        pw_probs = self.independent_decoder(tgt_embedding, y_mask, context, x_mask,
                                            dropout_params=None, trng=trng, use_noise=use_noise, softmax=False,
                                            context_info=context_info)

        # RNN Decoder - pass through the decoder conditional gru with attention
        hidden_decoder, context_decoder, opt_ret['dec_alphas'], _ = self.decoder(
            tgt_embedding, y_mask=y_mask, init_state=init_decoder_state, context=context, x_mask=x_mask,
            projected_context=pre_projected_context, dropout_params=dropout_params, one_step=False,
        )

        trng, use_noise, probs = self.get_word_probability(hidden_decoder, context_decoder, tgt_embedding,
                                                           trng=trng, use_noise=use_noise, pw_probs=pw_probs)

        rnn_test_cost = self.build_cost(y, y_mask, probs, epsilon=1e-6)
        pw_test_cost = self.build_cost(y, y_mask, pw_probs)
        test_cost = rnn_test_cost + pw_test_cost
        cost = test_cost / self.O['cost_normalization']

        return trng, use_noise, x, x_mask, y, y_mask, opt_ret, cost, test_cost, context_mean

    def build_sampler(self, **kwargs):
        batch_mode = kwargs.pop('batch_mode', False)
        trng = kwargs.pop('trng', RandomStreams(1234))
        use_noise = kwargs.pop('use_noise', theano.shared(np.float32(0.)))
        get_gates = kwargs.pop('get_gates', False)
        dropout_rate = kwargs.pop('dropout', False)

        # [NOTE] Fix the bug of BLEU drop here
        dropout_rate_out = 0.5 if 'fix_dp_bug' not in self.O or not self.O['fix_dp_bug'] else self.O['use_dropout']

        need_srcattn = kwargs.pop('need_srcattn', False)

        if dropout_rate is not False:
            dropout_params = [use_noise, trng, dropout_rate]
        else:
            dropout_params = None

        unit = self.O['unit']

        x = T.matrix('x', dtype='int64')
        xr = x[::-1]
        n_timestep = x.shape[0]
        n_samples = x.shape[1]

        if batch_mode:
            x_mask = T.matrix('x_mask', dtype=fX)
            xr_mask = x_mask[::-1]
        else:
            x_mask, xr_mask = None, None

        # Word embedding for forward rnn and backward rnn (source)
        src_embedding = self.embedding(x, n_timestep, n_samples)
        src_embedding_r = self.embedding(xr, n_timestep, n_samples)

        # Encoder
        ctx, _ = self.encoder(
            src_embedding, src_embedding_r,
            x_mask, xr_mask,
            dropout_params=dropout_params,
        )

        # Get the input for decoder rnn initializer mlp
        # TODO: change here to context_info (like build_model) or not?
        ctx_mean = self.get_context_mean(ctx, x_mask) if batch_mode else ctx.mean(0)
        init_state = self.feed_forward(ctx_mean, prefix='ff_state', activation=tanh)

        # Target embedding
        # TODO: add word embedding into position embedding?
        y_pos_ = T.repeat(T.arange(self.O['maxlen']).dimshuffle(0, 'x'), x.shape[0], 1)
        tgt_pos_embed = self.P['Wemb_dec_pos'][y_pos_.flatten()].reshape(
            [y_pos_.shape[0], y_pos_.shape[1], self.O['dim_word']])

        # Per-word prediction decoder.
        y_mask = T.alloc(floatX(1.), self.O['maxlen'], x.shape[0])

        # Output: probabilities of target words in this batch, ([Tt], [Bs], [V_tgt])
        pw_probs = self.independent_decoder(tgt_pos_embed, y_mask, ctx, x_mask,
                                            dropout_params=None, trng=trng, use_noise=use_noise, softmax=False)
        k = self.O['cond_softmax_k']
        top_k_args = theano_argpartsort(-pw_probs, k, axis=1)[:, :k]

        # Top-k vocabulary of this batch. ([Total vocab size of a batch],)
        top_k_vocab_tensor = theano_unique(top_k_args)
        # Shared variable for f_next. ([V_tgt],), 0-1 indicator
        top_k_vocab_indicator = theano.shared(np.empty([self.O['n_words']], dtype='int64'), name='top_k_vocab')

        print 'Building f_init...',
        inps = [x]
        if batch_mode:
            inps.append(x_mask)
        outs = [init_state, ctx]
        f_init = theano.function(
            inps, outs,
            name='f_init', profile=profile,
            updates=[
                (top_k_vocab_indicator, T.set_subtensor(T.zeros_like(top_k_vocab_indicator)[top_k_vocab_tensor], 1))
            ],
        )
        print 'Done'

        pre_projected_context_ = self.attention_projected_context(ctx, prefix='decoder')
        f_att_projected = theano.function([ctx], pre_projected_context_, name='f_att_projected', profile=profile)

        # x: 1 x 1
        y = T.vector('y_sampler', dtype='int64')
        init_state = T.tensor3('init_state', dtype=fX)
        init_memory = T.tensor3('init_memory', dtype=fX)

        t_indicator = theano.shared(np.int64(0), name='t_indicator')

        # If it's the first word, emb should be all zero and it is indicated by -1
        # word embedding + position embedding
        emb = T.switch(y[:, None] < 0,
                       T.alloc(0., 1, self.P['Wemb_dec'].shape[1]),
                       self.P['Wemb_dec'][y] + self.P['Wemb_dec_pos'][T.repeat(t_indicator, y.shape[0])])
        proj_ctx = T.tensor3('proj_ctx', dtype=fX)

        # Apply one step of conditional gru with attention
        hidden_decoder, context_decoder, alpha_src, kw_ret = self.decoder(
            emb, y_mask=None, init_state=init_state, context=ctx, projected_context=proj_ctx,
            x_mask=x_mask if batch_mode else None,
            dropout_params=dropout_params, one_step=True, init_memory=init_memory,
            get_gates=get_gates,
        )

        # Get memory_out and hiddens_without_dropout
        # FIXME: stack list into a single tensor
        memory_out = None
        hiddens_without_dropout = T.stack(kw_ret['hiddens_without_dropout'])
        if 'lstm' in unit:
            memory_out = T.stack(kw_ret['memory_outputs'])

        logit_lstm = self.feed_forward(hidden_decoder, prefix='ff_logit_lstm', activation=linear)
        logit_prev = self.feed_forward(emb, prefix='ff_logit_prev', activation=linear)
        logit_ctx = self.feed_forward(context_decoder, prefix='ff_logit_ctx', activation=linear)
        logit = T.tanh(logit_lstm + logit_prev + logit_ctx)

        if dropout_rate_out:
            logit = self.dropout(logit, use_noise, trng, dropout_rate_out)

        logit = self.feed_forward(logit, prefix='ff_logit', activation=linear)  # ([Bs], [V_tgt])

        row_index = T.arange(y.shape[0]).dimshuffle([0, 'x'])
        # Restore vocab from indicator.
        top_k_vocab = T.nonzero(top_k_vocab_indicator)[0]
        part_logit = logit[row_index, top_k_vocab]

        # Compute the softmax probability
        next_probs = T.nnet.softmax(part_logit)

        # Sample from softmax distribution to get the sample
        next_sample = top_k_vocab[trng.multinomial(pvals=next_probs).argmax(1)]

        # Compile a function to do the whole thing above, next word probability,
        # sampled word for the next target, next hidden state to be used
        print 'Building f_next..',
        inps = [y, ctx, proj_ctx, init_state]
        if batch_mode:
            inps.insert(2, x_mask)
        outs = [next_probs, next_sample, hiddens_without_dropout]
        if need_srcattn:
            outs.append(alpha_src)
        if 'lstm' in unit:
            inps.append(init_memory)
            outs.append(memory_out)
        if get_gates:
            outs.extend([
                T.stack(kw_ret['input_gates']),
                T.stack(kw_ret['forget_gates']),
                T.stack(kw_ret['output_gates']),
                kw_ret['input_gates_att'],
                kw_ret['forget_gates_att'],
                kw_ret['output_gates_att'],
            ])
        f_next = theano.function(
            inps, outs, name='f_next', profile=profile,
            updates=[(t_indicator, t_indicator + 1)]
        )
        print 'Done'

        return f_init, [f_next, f_att_projected]

    def gen_batch_sample(self, f_init, f_next, x, x_mask, trng=None, k=1, maxlen=30, eos_id=0, attn_src=False, **kwargs):
        return NMTModel.gen_batch_sample(self, f_init, f_next, x, x_mask, trng=trng, k=k, maxlen=maxlen, eos_id=eos_id,
                                         attn_src=attn_src, **kwargs)

    def stackLSTM(self, state_below, prefix='lstm', mask=None, context=None, context_mask=None, **kwargs):
        """
        Please note that in this setting, any layer before applying ctx_vec are non-linear transformations of position
        embedding. As as result, I think it is not of much practical value to ``multi-lstm'' the layers before ctx. The
        two possible structures are:
            (i) lstm + single attention
            (ii) lstm + multiple, even all attention
        I just implement (i) here
        """

        # TODO: change parameters

        dim = self.O['dim']
        assert context, 'Context must be provided'
        assert context.ndim == 3, 'Context must be 3-d: #annotation * #sample * dim'
        n_steps = state_below.shape[0]
        n_samples = state_below.shape[1]
        ctx_info = kwargs.get('context_info', None)
        if ctx_info is None:
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

    def independent_decoder(self, tgt_embedding, y_mask, context, x_mask, **kwargs):
        """

        Parameters
        ----------
        tgt_embedding:  ([Tt], [Bs], [W])
        y_mask:         ([Tt], [Bs])
        context:        ([Ts], [Bs], [Hc])
        x_mask:         ([Ts], [Bs])
        kwargs

        Returns
        -------
        ([Tt] * [Bs], [V_tgt])      # [V_tgt] = target vocab size = 30000
        """

        dim = self.O['dim']

        if self.O['delib_reversed'] in ('decoder', 'all'):
            context = context[::-1]
            x_mask = x_mask[::-1]

        if self.O['decoder_style'] == 'stackNN':
            projected_context = T.dot(context, self.P['attn_0_ctx2hidden'])     # projected_ctx: ([Ts], [Bs], [H])

            ctx_info = kwargs.get('context_info', None)
            if ctx_info is None:
                ctx_info = self.get_context_info(context, x_mask, tgt_embedding)

            H_ = T.dot(tgt_embedding, self.P['decoder_W_pose2h']) + \
                T.dot(ctx_info, self.P['decoder_W_ctx2h']) + self.P['decoder_b_i2h']
            for layer_id in xrange(self.O['n_decoder_layers']):
                H_ = T.tanh(H_)     # H_: ([Tt], [Bs], [n_in])

                # W: ([n_in], [H]); b: ([H])
                W = _slice(self.P[_p('decoder', 'W', layer_id)], 0, dim)
                b = self.P[_p('decoder', 'b', layer_id)][0 * dim: 1 * dim]

                if self.O['decoder_all_attention']:
                    # ctx_info: ([Tt], [Bs], [Hc])
                    ctx_info = self.attention_layer(context, x_mask, projected_context, H_, layer_id)
                    # <decoder_W_att2h>[layer_id]: ([Hc], [H])
                    H_ = T.dot(H_, W) + T.dot(ctx_info, self.P['decoder_W_att2h'][layer_id]) + b
                else:
                    H_ = T.dot(H_, W) + b
            H_ = T.tanh(H_)     # H_: ([Tt], [Bs], [H])

            trng = kwargs.pop('trng', RandomStreams(1234))
            use_noise = kwargs.pop('use_noise', theano.shared(np.float32(1.)))
            # [NOTE] Share logit parameters with LSTM
            logit_stackNN = self.feed_forward(H_, prefix='ff_logit_lstm', activation=linear)
            logit = T.tanh(logit_stackNN)
            if self.O['use_dropout']:
                logit = self.dropout(logit, use_noise, trng)
        elif self.O['decoder_style'] == 'stackLSTM':
            H_ = self.stackLSTM(tgt_embedding, 'decoder', y_mask, context, x_mask, **kwargs)
            logit = H_
        else:
            raise Exception('Not implemented yet')

        logit = self.feed_forward(logit, prefix='ff_logit', activation=linear)
        logit_shp = logit.shape
        unnormalized_probs = logit.reshape([-1, logit_shp[2]])
        if kwargs.pop('softmax', True):
            probs = T.nnet.softmax(unnormalized_probs)
            return probs
        else:
            return unnormalized_probs

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
            probs: Theano tensor variable, ([Tt] * [Bs], [V_tgt])
        """

        trng = kwargs.pop('trng', RandomStreams(1234))
        use_noise = kwargs.pop('use_noise', theano.shared(np.float32(0.)))

        logit_lstm = self.feed_forward(hidden_decoder, prefix='ff_logit_lstm', activation=linear)
        logit_prev = self.feed_forward(tgt_embedding, prefix='ff_logit_prev', activation=linear)
        logit_ctx = self.feed_forward(context_decoder, prefix='ff_logit_ctx', activation=linear)
        logit = T.tanh(logit_lstm + logit_prev + logit_ctx)  # n_timestep * n_sample * dim_word
        if self.O['dropout_out']:
            logit = self.dropout(logit, use_noise, trng, self.O['dropout_out'])

        # n_timestep * n_sample * n_words
        logit = self.feed_forward(logit, prefix='ff_logit', activation=linear)
        logit_shp = logit.shape
        logit_reshaped = logit.reshape([logit_shp[0] * logit_shp[1], logit_shp[2]])

        probs = self._conditional_softmax(logit_reshaped, kwargs.pop('pw_probs', None))

        return trng, use_noise, probs

    def _conditional_softmax(self, logit_2d, pw_probs=None):
        """
        Apply conditional softmax on probabilities.

        Parameters
        ----------
        logit_2d: Theano tensor variable, 2D-array
            Probabilities from RNN decoder.
            Shape: ([Tt] * [Bs], [V_tgt]) in training stage
        pw_probs: Theano tensor variable, 2D-array
            Probabilities (unnormalized) from the per-word prediction decoder.
            Shape: ([Tt] * [Bs], [V_tgt]) in training stage
        Returns
        -------
        probs: Theano tensor variable, 2D-array
            Conditional softmax result
            Shape: same as logit_2d
        Notes
        -----
        Only run in training stage
        """

        k = self.O['cond_softmax_k']

        if pw_probs is not None:
            top_k_args = theano_argpartsort(-pw_probs, k, axis=1)[:, :k]    # top_k_args: ([Tt] * [Bs], k)

            # [NOTE] indices to get/set top-k value
            top_k_indices = (T.arange(pw_probs.shape[0]).dimshuffle([0, 'x']), top_k_args)

            probs = T.alloc(floatX(0.), *logit_2d.shape)
            # get top-k probability indices from per-word probs, then apply it into probs
            probs = T.set_subtensor(probs[top_k_indices], T.nnet.softmax(logit_2d[top_k_indices]))
        else:
            probs = T.nnet.softmax(logit_2d)

        return probs
