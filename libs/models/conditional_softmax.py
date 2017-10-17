#! /usr/bin/python
# -*- coding: utf-8 -*-

from collections import OrderedDict
from contextlib import contextmanager

import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from .deliberation import DelibNMT, DelibInitializer, NMTModel
from ..layers import *
from ..constants import fX, profile
from ..utility.basic import floatX, arg_top_k
from ..utility.utils import _p, debug_print

__author__ = 'fyabc'


class ArgPartSortOp(theano.Op):
    """
    This class is a wrapper of numpy bottleneck arg-top-k function.

    """

    __props__ = ('k',)

    def __init__(self, k):
        self.k = k

    def __str__(self):
        return self.__class__.__name__ + '{%d}' % self.k

    def make_node(self, input_, axis=-1):
        input_ = T.as_tensor_variable(input_)
        axis = T.as_tensor_variable(axis)
        bcast = input_.type.broadcastable
        return theano.Apply(self, [input_, axis], [T.TensorType(dtype='int64', broadcastable=bcast)()])

    def perform(self, node, inputs, output_storage, params=None):
        a = inputs[0]
        axis = inputs[1]
        z = output_storage[0]
        z[0] = theano._asarray(arg_top_k(a, self.k, axis), dtype=node.outputs[0].dtype)

    def infer_shape(self, node, inputs_shapes):
        if (isinstance(node.inputs[1], theano.Constant) and
                node.inputs[1].data is None):
            return [(T.mul(*inputs_shapes[0]),)]
        # axis should not be None, so there should be the same number of
        # dimensions in the input and output
        assert node.inputs[0].ndim == node.outputs[0].ndim
        assert inputs_shapes[1] == ()
        return [inputs_shapes[0]]

    def grad(self, inputs, outputs_grads):
        # No grad defined for integers.
        inp, axis = inputs
        inp_grad = inp.zeros_like()
        axis_grad = theano.gradient.grad_undefined(
            self, 1, axis,
            "argpartsort is not defined for non-integer axes so"
            " argpartsort(x, axis+eps) is undefined")
        return [inp_grad, axis_grad]

    def R_op(self, inputs, eval_points):
        raise NotImplementedError('R-op not implemented for argpartsort')


def theano_argpartsort(a, k, axis=-1):
    if axis is None:
        a = a.flatten()
        axis = 0
    return ArgPartSortOp(k)(a, axis)


@contextmanager
def _delib_env(obj, keys=None):
    """Context manager to set the per-word prediction options `obj.DO` as options.

    Or only set some keys.
    """

    if isinstance(keys, str):
        keys = [keys]

    if keys:
        tmp_o = {}
        for key in keys:
            tmp_o[key] = obj.O[key]
            obj.O[key] = obj.DO[key]
    else:
        tmp_o = obj.O
        obj.O = obj.DO

    try:
        yield tmp_o
    finally:
        if keys:
            for key in keys:
                obj.DO[key] = obj.O[key]
                obj.O[key] = tmp_o[key]
        else:
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
        # Not used now.
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

        to_be_deleted.add('Wemb_dec_pos')

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
        with _delib_env(self, 'use_src_pos'):
            (x, x_mask, y, y_mask), context, _ = self.input_to_context(dropout_params=dropout_params)

        # Per-word prediction decoder.
        with _delib_env(self):
            y_pos_ = T.repeat(T.arange(y.shape[0]).dimshuffle(0, 'x'), y.shape[1], 1)
            tgt_pos_embed = self.P['Wemb_dec_pos'][y_pos_.flatten()].reshape(
                [y_pos_.shape[0], y_pos_.shape[1], self.DO['dim_word']])
            pw_probs = self.independent_decoder(tgt_pos_embed, y, y_mask, context, x_mask,
                                                dropout_params=None, trng=trng, use_noise=use_noise)

        # RNN decoder.
        n_timestep, n_timestep_tgt, n_samples = self.input_dimensions(x, y)

        context_mean = self.get_context_mean(context, x_mask)
        # Initial decoder state
        init_decoder_state = self.feed_forward(context_mean, prefix='ff_state', activation=tanh)

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
        test_cost = self.build_cost(y, y_mask, probs, epsilon=1e-6)
        cost = test_cost / self.O['cost_normalization']  # cost used to derive gradient in training

        return trng, use_noise, x, x_mask, y, y_mask, opt_ret, cost, test_cost, self._ctx_info

    def build_sampler(self, **kwargs):
        batch_mode = kwargs.pop('batch_mode', False)
        trng = kwargs.pop('trng', RandomStreams(1234))
        use_noise = kwargs.pop('use_noise', theano.shared(np.float32(0.)))
        get_gates = kwargs.pop('get_gates', False)
        dropout_rate = kwargs.pop('dropout', False)
        dropout_rate_out = self.O['dropout_out']
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

        # Word embedding for forward rnn and backward rnn (source)
        src_embedding = self.embedding(x, n_timestep, n_samples)
        src_embedding_r = self.embedding(xr, n_timestep, n_samples)

        # Encoder
        ctx, _ = self.encoder(
            src_embedding, src_embedding_r,
            x_mask if batch_mode else None, xr_mask if batch_mode else None,
            dropout_params=dropout_params,
        )

        # Get the input for decoder rnn initializer mlp
        ctx_mean = self.get_context_mean(ctx, x_mask) if batch_mode else ctx.mean(0)
        init_state = self.feed_forward(ctx_mean, prefix='ff_state', activation=tanh)

        print 'Building f_init...',
        inps = [x]
        if batch_mode:
            inps.append(x_mask)
        outs = [init_state, ctx]
        f_init = theano.function(inps, outs, name='f_init', profile=profile,)
        print 'Done'

        pre_projected_context_ = self.attention_projected_context(ctx, prefix='decoder')
        f_att_projected = theano.function([ctx], pre_projected_context_, name='f_att_projected', profile=profile)

        # x: 1 x 1
        y = T.vector('y_sampler', dtype='int64')
        init_state = T.tensor3('init_state', dtype=fX)
        init_memory = T.tensor3('init_memory', dtype=fX)

        t_indicator = theano.shared(np.int64(0), name='t_indicator')

        # If it's the first word, emb should be all zero and it is indicated by -1
        emb = T.switch(y[:, None] < 0,
                       T.alloc(0., 1, self.P['Wemb_dec'].shape[1]),
                       self.P['Wemb_dec'][y])
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

        logit = self.feed_forward(logit, prefix='ff_logit', activation=linear)

        logit, _ = debug_print(logit, '$ logit:')

        # Per-word prediction decoder.
        # todo: fix shape mismatch bug of pw_probs and logits in test mode
        with _delib_env(self):
            y_ = None
            # y_pos_ = T.repeat(T.arange(self.O['maxlen']).dimshuffle(0, 'x'), y.shape[0], 1)
            # y_mask = T.alloc(floatX(1.), self.O['maxlen'], y.shape[0])
            y_pos_ = T.alloc(t_indicator, 1, y.shape[0])
            y_mask = T.alloc(floatX(1.), 1, y.shape[0])

            x_mask, _ = debug_print(x_mask, '$ x_mask:')

            tgt_pos_embed = self.P['Wemb_dec_pos'][y_pos_.flatten()].reshape(
                [y_pos_.shape[0], y_pos_.shape[1], self.O['dim_word']])
            # todo: fix `x_mask` for non-batch mode
            pw_probs = self.independent_decoder(tgt_pos_embed, y_, y_mask, ctx, x_mask,
                                                dropout_params=None, trng=trng, use_noise=use_noise)
        # with _delib_env(self):
        #     # [NOTE]: In testing stage of per-word prediction decoder,
        #     # y: (1, [Bs]); y_pos_: (1, [Bs]), y_mask: (1, [Bs])
        #     # different from RNN decoder one-step mode.
        #     y_ = y.dimshuffle(['x', 0])
        #     y_pos_ = T.repeat(t_indicator, y.shape[0]).dimshuffle(['x', 0])
        #     y_mask = T.alloc(floatX(1.), 1, y.shape[0])
        #
        #     tgt_pos_embed = self.P['Wemb_dec_pos'][y_pos_.flatten()].reshape(
        #         [y_pos_.shape[0], y_pos_.shape[1], self.O['dim_word']])
        #     pw_probs = self.independent_decoder(tgt_pos_embed, y_, y_mask, ctx, x_mask,
        #                                         dropout_params=None, trng=trng, use_noise=use_noise)

        # Compute the softmax probability
        next_probs = self._conditional_softmax(logit, pw_probs=pw_probs)

        # Sample from softmax distribution to get the sample
        next_sample = trng.multinomial(pvals=next_probs).argmax(1)

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
            updates={t_indicator: t_indicator + 1}
        )
        print 'Done'

        return f_init, [f_next, f_att_projected]

    def gen_batch_sample(self, f_init, f_next, x, x_mask, trng=None, k=1, maxlen=30, eos_id=0, attn_src=False, **kwargs):
        return NMTModel.gen_batch_sample(self, f_init, f_next, x, x_mask, trng=trng, k=k, maxlen=maxlen, eos_id=eos_id,
                                         attn_src=attn_src, **kwargs)

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
            probs: Theano tensor variable, ([Tt] * [Bs], n_words)
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
            Shape: ([Tt] * [Bs], n_words) in training stage, ([lived_Bs], n_words) in testing stage
        pw_probs: Theano tensor variable, 2D-array
            Probabilities (unnormalized) from the per-word prediction decoder.
            Shape: ([Tt] * [Bs], n_words) in training stage, (???, n_words) in testing stage
        Returns
        -------
        probs: Theano tensor variable, 2D-array
            Conditional softmax result
            Shape: same as logit_2d
        """

        k = self.O['cond_softmax_k']

        if pw_probs is not None:
            # top_k_args = T.argsort(-pw_probs)[:, :k]
            top_k_args = theano_argpartsort(-pw_probs, k, axis=1)[:, :k]
            # [NOTE] indices to get/set top-k value
            top_k_indices = (T.arange(pw_probs.shape[0]).dimshuffle([0, 'x']), top_k_args)

            probs = T.alloc(floatX(0.), *pw_probs.shape)
            # get top-k probability indices from per-word probs, then apply it into probs
            probs = T.set_subtensor(probs[top_k_indices], T.nnet.softmax(logit_2d[top_k_indices]))
        else:
            probs = T.nnet.softmax(logit_2d)

        return probs
