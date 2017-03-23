#! /usr/bin/python
# -*- coding: utf-8 -*-

from collections import OrderedDict

import numpy as np
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from constants import fX, profile
from utils import _p, normal_weight, orthogonal_weight, concatenate

__author__ = 'fyabc'


# Some utilities.

def _slice(_x, n, dim):
    """Utility function to slice a tensor."""
    if _x.ndim == 3:
        return _x[:, :, n * dim:(n + 1) * dim]
    return _x[:, n * dim:(n + 1) * dim]


# Activations.

def tanh(x):
    return T.tanh(x)


def linear(x):
    return x


# Some helper layers.

def dropout_layer(state_before, use_noise, trng, dropout_rate=0.5):
    """Dropout"""

    projection = T.switch(
        use_noise,
        state_before * trng.binomial(state_before.shape, p=(1. - dropout_rate), n=1,
                                     dtype=state_before.dtype),
        state_before * (1. - dropout_rate))
    return projection


def embedding(tparams, state_below, O, n_timestep, n_samples):
    """Embedding"""

    emb = tparams['Wemb'][state_below.flatten()]
    emb = emb.reshape([n_timestep, n_samples, O['dim_word']])

    return emb


def attention_layer(context_mask, et, ht_1, We_att, Wh_att, Wb_att, U_att, Ub_att):
    """Attention"""

    a_network = T.tanh(T.dot(et, We_att) + T.dot(ht_1, Wh_att) + Wb_att)
    alpha = T.dot(a_network, U_att) + Ub_att
    alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
    alpha = T.exp(alpha)
    if context_mask:
        alpha *= context_mask
    alpha = alpha / alpha.sum(0, keepdims=True)
    # if Wp_compress_e:
    #    ctx_t = (tensor.dot(et, Wp_compress_e) * alpha[:,:,None]).sum(0) # This is the c_t in Baidu's paper
    # else:
    #    ctx_t = (et * alpha[:,:,None]).sum(0)
    ctx_t = (et * alpha[:, :, None]).sum(0)
    return ctx_t


def param_init_fflayer(O, params, prefix='ff', nin=None, nout=None,
                       orthogonal=True):
    """feedforward layer: affine transformation + point-wise nonlinearity"""

    if nin is None:
        nin = O['dim_proj']
    if nout is None:
        nout = O['dim_proj']
    params[_p(prefix, 'W')] = normal_weight(nin, nout, scale=0.01, orthogonal=orthogonal)
    params[_p(prefix, 'b')] = np.zeros((nout,), dtype=fX)

    return params


def fflayer(tparams, state_below, O, prefix='rconv',
            activ=tanh, **kwargs):
    if isinstance(activ, (str, unicode)):
        activ = eval(activ)
    return activ(T.dot(state_below, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')])


def param_init_gru(O, params, prefix='gru', nin=None, dim=None, **kwargs):
    if nin is None:
        nin = O['dim_proj']
    if dim is None:
        dim = O['dim_proj']

    layer_id = kwargs.pop('layer_id', 0)

    # embedding to gates transformation weights, biases
    W = np.concatenate([normal_weight(nin, dim), normal_weight(nin, dim)], axis=1)
    params[_p(prefix, 'W', layer_id)] = W
    params[_p(prefix, 'b', layer_id)] = np.zeros((2 * dim,), dtype=fX)

    # recurrent transformation weights for gates
    U = np.concatenate([orthogonal_weight(dim),
                        orthogonal_weight(dim)], axis=1)
    params[_p(prefix, 'U', layer_id)] = U

    # embedding to hidden state proposal weights, biases
    Wx = normal_weight(nin, dim)
    params[_p(prefix, 'Wx', layer_id)] = Wx
    params[_p(prefix, 'bx', layer_id)] = np.zeros((dim,), dtype=fX)

    # recurrent transformation weights for hidden state proposal
    Ux = orthogonal_weight(dim)
    params[_p(prefix, 'Ux', layer_id)] = Ux

    return params


def _gru_step_slice(src_mask, x_, xx_, ht_1, U, Ux):
    """GRU step function to be used by scan

    arguments (0) | sequences (3) | outputs-info (1) | non-seqs (2)
    """

    _dim = Ux.shape[1]

    preact = T.dot(ht_1, U) + x_

    # reset and update gates
    r = T.nnet.sigmoid(_slice(preact, 0, _dim))
    u = T.nnet.sigmoid(_slice(preact, 1, _dim))

    # hidden state proposal
    ht_tilde = T.tanh(T.dot(ht_1, Ux) * r + xx_)

    # leaky integrate and obtain next hidden state
    ht = u * ht_1 + (1. - u) * ht_tilde
    ht = src_mask[:, None] * ht + (1. - src_mask)[:, None] * ht_1

    return ht


def gru_layer(tparams, state_below, O, prefix='gru', mask=None, **kwargs):
    """GRU layer"""

    layer_id = kwargs.pop('layer_id', 0)
    dropout_params = kwargs.pop('dropout_params', None)

    n_steps = state_below.shape[0]
    n_samples = state_below.shape[1] if state_below.ndim == 3 else 1

    dim = tparams[_p(prefix, 'Ux', layer_id)].shape[1]

    if mask is None:
        mask = T.alloc(1., state_below.shape[0], 1)

    # state_below is the input word embeddings
    # input to the gates, concatenated
    state_below_ = T.dot(state_below, tparams[_p(prefix, 'W', layer_id)]) + tparams[_p(prefix, 'b', layer_id)]
    # input to compute the hidden state proposal
    state_belowx = T.dot(state_below, tparams[_p(prefix, 'Wx', layer_id)]) + tparams[_p(prefix, 'bx', layer_id)]

    # prepare scan arguments
    seqs = [mask, state_below_, state_belowx]
    init_states = [T.alloc(0., n_samples, dim)]
    _step = _gru_step_slice
    shared_vars = [tparams[_p(prefix, 'U', layer_id)],
                   tparams[_p(prefix, 'Ux', layer_id)]]

    outputs, updates = theano.scan(
        _step,
        sequences=seqs,
        outputs_info=init_states,
        non_sequences=shared_vars,
        name=_p(prefix, '_layers', layer_id),
        n_steps=n_steps,
        profile=profile,
        strict=True,
    )

    if dropout_params:
        outputs = dropout_layer(outputs, *dropout_params)

    # [NOTE] Be compatible with GRU conditional layer
    outputs = [outputs]

    return outputs


def param_init_gru_cond(O, params, prefix='gru_cond', nin=None, dim=None, dimctx=None, nin_nonlin=None,
                        dim_nonlin=None):
    if nin is None:
        nin = O['dim']
    if dim is None:
        dim = O['dim']
    if dimctx is None:
        dimctx = O['dim']
    if nin_nonlin is None:
        nin_nonlin = nin
    if dim_nonlin is None:
        dim_nonlin = dim

    W = np.concatenate([normal_weight(nin, dim),
                        normal_weight(nin, dim)], axis=1)
    params[_p(prefix, 'W')] = W
    params[_p(prefix, 'b')] = np.zeros((2 * dim,), dtype=fX)
    U = np.concatenate([orthogonal_weight(dim_nonlin),
                        orthogonal_weight(dim_nonlin)], axis=1)
    params[_p(prefix, 'U')] = U

    Wx = normal_weight(nin_nonlin, dim_nonlin)
    params[_p(prefix, 'Wx')] = Wx
    Ux = orthogonal_weight(dim_nonlin)
    params[_p(prefix, 'Ux')] = Ux
    params[_p(prefix, 'bx')] = np.zeros((dim_nonlin,), dtype=fX)

    U_nl = np.concatenate([orthogonal_weight(dim_nonlin),
                           orthogonal_weight(dim_nonlin)], axis=1)
    params[_p(prefix, 'U_nl')] = U_nl
    params[_p(prefix, 'b_nl')] = np.zeros((2 * dim_nonlin,), dtype=fX)

    Ux_nl = orthogonal_weight(dim_nonlin)
    params[_p(prefix, 'Ux_nl')] = Ux_nl
    params[_p(prefix, 'bx_nl')] = np.zeros((dim_nonlin,), dtype=fX)

    # context to LSTM
    Wc = normal_weight(dimctx, dim * 2)
    params[_p(prefix, 'Wc')] = Wc

    Wcx = normal_weight(dimctx, dim)
    params[_p(prefix, 'Wcx')] = Wcx

    # attention: combined -> hidden
    W_comb_att = normal_weight(dim, dimctx)
    params[_p(prefix, 'W_comb_att')] = W_comb_att

    # attention: context -> hidden
    Wc_att = normal_weight(dimctx)
    params[_p(prefix, 'Wc_att')] = Wc_att

    # attention: hidden bias
    b_att = np.zeros((dimctx,), dtype=fX)
    params[_p(prefix, 'b_att')] = b_att

    # attention:
    U_att = normal_weight(dimctx, 1)
    params[_p(prefix, 'U_att')] = U_att
    c_att = np.zeros((1,), dtype=fX)
    params[_p(prefix, 'c_tt')] = c_att

    return params


def gru_cond_layer(tparams, state_below, O, prefix='gru', mask=None, context=None, one_step=False, init_memory=None,
                   init_state=None, context_mask=None, **kwargs):
    """Conditional GRU layer with Attention"""

    assert context, 'Context must be provided'

    if one_step:
        assert init_state, 'previous state must be provided'

    n_steps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    # mask
    if mask is None:
        mask = T.alloc(1., state_below.shape[0], 1)

    dim = tparams[_p(prefix, 'Wcx')].shape[1]

    # initial/previous state
    if init_state is None:
        init_state = T.alloc(0., n_samples, dim)

    # projected context
    assert context.ndim == 3, \
        'Context must be 3-d: #annotation x #sample x dim'
    pctx_ = T.dot(context, tparams[_p(prefix, 'Wc_att')]) + \
            tparams[_p(prefix, 'b_att')]

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    # projected x
    state_belowx = T.dot(state_below, tparams[_p(prefix, 'Wx')]) + tparams[_p(prefix, 'bx')]
    state_below_ = T.dot(state_below, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')]

    def _step_slice(m_, x_, xx_,
                    h_, ctx_, alpha_, pctx_, cc_,
                    U, Wc, W_comb_att, U_att, c_tt, Ux, Wcx, U_nl, Ux_nl, b_nl, bx_nl):
        preact1 = T.dot(h_, U)
        preact1 += x_
        preact1 = T.nnet.sigmoid(preact1)

        r1 = _slice(preact1, 0, dim)
        u1 = _slice(preact1, 1, dim)

        preactx1 = T.dot(h_, Ux)
        preactx1 *= r1
        preactx1 += xx_

        h1 = T.tanh(preactx1)

        h1 = u1 * h_ + (1. - u1) * h1
        h1 = m_[:, None] * h1 + (1. - m_)[:, None] * h_

        # attention
        pstate_ = T.dot(h1, W_comb_att)
        pctx__ = pctx_ + pstate_[None, :, :]
        # pctx__ += xc_
        pctx__ = T.tanh(pctx__)
        alpha = T.dot(pctx__, U_att) + c_tt
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
        alpha = T.exp(alpha)
        if context_mask:
            alpha = alpha * context_mask
        alpha = alpha / alpha.sum(0, keepdims=True)
        ctx_ = (cc_ * alpha[:, :, None]).sum(0)  # current context

        preact2 = T.dot(h1, U_nl) + b_nl
        preact2 += T.dot(ctx_, Wc)
        preact2 = T.nnet.sigmoid(preact2)

        r2 = _slice(preact2, 0, dim)
        u2 = _slice(preact2, 1, dim)

        preactx2 = T.dot(h1, Ux_nl) + bx_nl
        preactx2 *= r2
        preactx2 += T.dot(ctx_, Wcx)

        h2 = T.tanh(preactx2)

        h2 = u2 * h1 + (1. - u2) * h2
        h2 = m_[:, None] * h2 + (1. - m_)[:, None] * h1

        return h2, ctx_, alpha.T  # pstate_, preact, preactx, r, u

    seqs = [mask, state_below_, state_belowx]
    # seqs = [mask, state_below_, state_belowx, state_belowc]
    _step = _step_slice

    shared_vars = [tparams[_p(prefix, 'U')],
                   tparams[_p(prefix, 'Wc')],
                   tparams[_p(prefix, 'W_comb_att')],
                   tparams[_p(prefix, 'U_att')],
                   tparams[_p(prefix, 'c_tt')],
                   tparams[_p(prefix, 'Ux')],
                   tparams[_p(prefix, 'Wcx')],
                   tparams[_p(prefix, 'U_nl')],
                   tparams[_p(prefix, 'Ux_nl')],
                   tparams[_p(prefix, 'b_nl')],
                   tparams[_p(prefix, 'bx_nl')]]

    if one_step:
        rval = _step(*(seqs + [init_state, None, None, pctx_, context] +
                       shared_vars))
    else:
        rval, updates = theano.scan(_step,
                                    sequences=seqs,
                                    outputs_info=[init_state,
                                                  T.alloc(0., n_samples,
                                                          context.shape[2]),
                                                  T.alloc(0., n_samples,
                                                          context.shape[0])],
                                    non_sequences=[pctx_, context] + shared_vars,
                                    name=_p(prefix, '_layers'),
                                    n_steps=n_steps,
                                    profile=profile,
                                    strict=True)
    return rval


def gru_encoder(tparams, src_embedding, src_embedding_r, x_mask, xr_mask, O, dropout_params=None):
    """Multi-layer GRU encoder.

    :return Context vector
    """

    global_f = src_embedding
    global_f_r = src_embedding_r

    # Hidden states of last layer
    h_last = None
    h_last_r = None

    for layer_id in xrange(O['n_encoder_layers']):
        if layer_id > 0:
            # [NOTE] Add more connections (fast-forward, highway, ...) here
            if True:
                global_f = h_last
                global_f_r = h_last_r

        h_last = get_build(O['encoder'])(tparams, global_f, O, prefix='encoder', mask=x_mask, layer_id=layer_id,
                                         dropout_params=dropout_params)[0]
        h_last_r = get_build(O['encoder'])(tparams, global_f_r, O, prefix='encoder_r', mask=xr_mask, layer_id=layer_id,
                                           dropout_params=dropout_params)[0]

    # context will be the concatenation of forward and backward rnns
    context = concatenate([h_last, h_last_r[::-1]], axis=h_last.ndim - 1)

    return context


# layers: 'name': ('parameter initializer', 'feedforward')
layers = {
    'ff': (param_init_fflayer, fflayer),
    'gru': (param_init_gru, gru_layer),
    'gru_cond': (param_init_gru_cond, gru_cond_layer),
}


def get_layer(name):
    fns = layers[name]
    return fns[0], fns[1]


def get_init(name):
    return layers[name][0]


def get_build(name):
    return layers[name][1]


def init_params(O):
    """Initialize all parameters"""

    params = OrderedDict()

    # embedding
    params['Wemb'] = normal_weight(O['n_words_src'], O['dim_word'])
    params['Wemb_dec'] = normal_weight(O['n_words'], O['dim_word'])

    # encoder: bidirectional RNN
    for layer_id in xrange(O['n_encoder_layers']):
        params = get_init(O['encoder'])(O, params, prefix='encoder', nin=O['dim_word'], dim=O['dim'], layer_id=layer_id)
        params = get_init(O['encoder'])(O, params, prefix='encoder_r', nin=O['dim_word'], dim=O['dim'],
                                        layer_id=layer_id)

    context_dim = 2 * O['dim']
    # init_state, init_cell
    params = get_init('ff')(O, params, prefix='ff_state', nin=context_dim, nout=O['dim'])
    # decoder
    params = get_init(O['decoder'])(O, params, prefix='decoder', nin=O['dim_word'], dim=O['dim'], dimctx=context_dim)
    # readout
    params = get_init('ff')(O, params, prefix='ff_logit_lstm', nin=O['dim'], nout=O['dim_word'], orthogonal=False)
    params = get_init('ff')(O, params, prefix='ff_logit_prev', nin=O['dim_word'], nout=O['dim_word'], orthogonal=False)
    params = get_init('ff')(O, params, prefix='ff_logit_ctx', nin=context_dim, nout=O['dim_word'], orthogonal=False)
    params = get_init('ff')(O, params, prefix='ff_logit', nin=O['dim_word'], nout=O['n_words'])

    return params


def build_model(tparams, O):
    """Build a training model."""

    opt_ret = dict()

    trng = RandomStreams(1234)
    use_noise = theano.shared(np.float32(0.))

    # description string: #words * #samples
    x = T.matrix('x', dtype='int64')
    x_mask = T.matrix('x_mask', dtype=fX)
    y = T.matrix('y', dtype='int64')
    y_mask = T.matrix('y_mask', dtype=fX)

    # for the backward rnn, we just need to invert x and x_mask
    xr = x[::-1]
    xr_mask = x_mask[::-1]

    n_timestep = x.shape[0]
    n_timestep_tgt = y.shape[0]
    n_samples = x.shape[1]

    # word embedding for forward rnn and backward rnn (source)
    src_embedding = embedding(tparams, x, O, n_timestep, n_samples)
    src_embedding_r = embedding(tparams, xr, O, n_timestep, n_samples)

    ctx = gru_encoder(tparams, src_embedding, src_embedding_r, x_mask, xr_mask, O, dropout_params=None)

    # mean of the context (across time) will be used to initialize decoder rnn
    ctx_mean = (ctx * x_mask[:, :, None]).sum(0) / x_mask.sum(0)[:, None]

    # or you can use the last state of forward + backward encoder rnns
    # ctx_mean = concatenate([proj[0][-1], projr[0][-1]], axis=proj[0].ndim-2)

    # initial decoder state
    init_state = get_build('ff')(tparams, ctx_mean, O, prefix='ff_state', activ=tanh)

    # word embedding (target), we will shift the target sequence one time step
    # to the right. This is done because of the bi-gram connections in the
    # readout and decoder rnn. The first target will be all zeros and we will
    # not condition on the last output.
    tgt_embedding = tparams['Wemb_dec'][y.flatten()]
    tgt_embedding = tgt_embedding.reshape([n_timestep_tgt, n_samples, O['dim_word']])
    emb_shifted = T.zeros_like(tgt_embedding)
    emb_shifted = T.set_subtensor(emb_shifted[1:], tgt_embedding[:-1])
    tgt_embedding = emb_shifted

    # decoder - pass through the decoder conditional gru with attention
    proj = get_build(O['decoder'])(tparams, tgt_embedding, O, prefix='decoder', mask=y_mask, context=ctx,
                                   context_mask=x_mask, one_step=False, init_state=init_state)
    # hidden states of the decoder gru
    proj_h = proj[0]  # n_timestep * n_sample * dim

    # weighted averages of context, generated by attention module
    ctxs = proj[1]

    # weights (alignment matrix)
    opt_ret['dec_alphas'] = proj[2]

    # compute word probabilities
    logit_lstm = get_build('ff')(tparams, proj_h, O, prefix='ff_logit_lstm', activ='linear')
    logit_prev = get_build('ff')(tparams, tgt_embedding, O, prefix='ff_logit_prev', activ='linear')
    logit_ctx = get_build('ff')(tparams, ctxs, O, prefix='ff_logit_ctx', activ='linear')
    logit = T.tanh(logit_lstm + logit_prev + logit_ctx)  # n_timestep * n_sample * dim_word
    if O['use_dropout']:
        logit = dropout_layer(logit, use_noise, trng)
    logit = get_build('ff')(tparams, logit, O, prefix='ff_logit', activ='linear')  # n_timestep * n_sample * n_words
    logit_shp = logit.shape
    probs = T.nnet.softmax(logit.reshape([logit_shp[0] * logit_shp[1],
                                          logit_shp[2]]))

    # cost
    y_flat = y.flatten()
    y_flat_idx = T.arange(y_flat.shape[0]) * O['n_words'] + y_flat
    cost = -T.log(probs.flatten()[y_flat_idx])
    cost = cost.reshape([y.shape[0], y.shape[1]])
    cost = (cost * y_mask).sum(0)

    if O['plot_graph'] is not None:
        print 'Plotting pre-compile graph...',
        theano.printing.pydotprint(
            cost,
            outfile='pictures/pre_compile_{}'.format(O['plot_graph']),
            var_with_name_simple=True,
        )
        print 'Done'

    return trng, use_noise, x, x_mask, y, y_mask, opt_ret, cost, ctx_mean


def build_sampler(tparams, O, trng, use_noise):
    """Build a sampler."""

    x = T.matrix('x', dtype='int64')
    xr = x[::-1]
    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    # word embedding (source), forward and backward
    emb = tparams['Wemb'][x.flatten()]
    emb = emb.reshape([n_timesteps, n_samples, O['dim_word']])
    embr = tparams['Wemb'][xr.flatten()]
    embr = embr.reshape([n_timesteps, n_samples, O['dim_word']])

    # encoder
    proj = get_build(O['encoder'])(tparams, emb, O,
                                   prefix='encoder')
    projr = get_build(O['encoder'])(tparams, embr, O,
                                    prefix='encoder_r')

    # concatenate forward and backward rnn hidden states
    ctx = concatenate([proj[0], projr[0][::-1]], axis=proj[0].ndim - 1)

    # get the input for decoder rnn initializer mlp
    ctx_mean = ctx.mean(0)
    # ctx_mean = concatenate([proj[0][-1],projr[0][-1]], axis=proj[0].ndim-2)
    init_state = get_build('ff')(tparams, ctx_mean, O,
                                 prefix='ff_state', activ='tanh')

    print 'Building f_init...',
    outs = [init_state, ctx]
    f_init = theano.function([x], outs, name='f_init', profile=profile)
    print 'Done'

    # x: 1 x 1
    y = T.vector('y_sampler', dtype='int64')
    init_state = T.matrix('init_state', dtype=fX)

    # if it's the first word, emb should be all zero and it is indicated by -1
    emb = T.switch(y[:, None] < 0,
                   T.alloc(0., 1, tparams['Wemb_dec'].shape[1]),
                   tparams['Wemb_dec'][y])

    # apply one step of conditional gru with attention
    proj = get_build(O['decoder'])(tparams, emb, O,
                                   prefix='decoder',
                                   mask=None, context=ctx,
                                   one_step=True,
                                   init_state=init_state)
    # get the next hidden state
    next_state = proj[0]

    # get the weighted averages of context for this target word y
    ctxs = proj[1]

    logit_lstm = get_build('ff')(tparams, next_state, O,
                                 prefix='ff_logit_lstm', activ='linear')
    logit_prev = get_build('ff')(tparams, emb, O,
                                 prefix='ff_logit_prev', activ='linear')
    logit_ctx = get_build('ff')(tparams, ctxs, O,
                                prefix='ff_logit_ctx', activ='linear')
    logit = T.tanh(logit_lstm + logit_prev + logit_ctx)
    if O['use_dropout']:
        logit = dropout_layer(logit, use_noise, trng)
    logit = get_build('ff')(tparams, logit, O,
                            prefix='ff_logit', activ='linear')

    # compute the softmax probability
    next_probs = T.nnet.softmax(logit)

    # sample from softmax distribution to get the sample
    next_sample = trng.multinomial(pvals=next_probs).argmax(1)

    # compile a function to do the whole thing above, next word probability,
    # sampled word for the next target, next hidden state to be used
    print 'Building f_next..',
    inps = [y, ctx, init_state]
    outs = [next_probs, next_sample, next_state]
    f_next = theano.function(inps, outs, name='f_next', profile=profile)
    print 'Done'

    return f_init, f_next
