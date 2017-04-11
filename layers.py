#! /usr/bin/python
# -*- coding: utf-8 -*-

from collections import OrderedDict

import numpy as np
import theano
from theano import tensor as T

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


def param_init_feed_forward(O, params, prefix='ff', nin=None, nout=None,
                            orthogonal=True):
    """feedforward layer: affine transformation + point-wise nonlinearity"""

    if nin is None:
        nin = O['dim_proj']
    if nout is None:
        nout = O['dim_proj']
    params[_p(prefix, 'W')] = normal_weight(nin, nout, scale=0.01, orthogonal=orthogonal)
    params[_p(prefix, 'b')] = np.zeros((nout,), dtype=fX)

    return params


def feed_forward(P, state_below, O, prefix='rconv',
                 activ=tanh, **kwargs):
    if isinstance(activ, (str, unicode)):
        activ = eval(activ)
    return activ(T.dot(state_below, P[_p(prefix, 'W')]) + P[_p(prefix, 'b')])


def param_init_gru(O, params, prefix='gru', nin=None, dim=None, **kwargs):
    if nin is None:
        nin = O['dim_proj']
    if dim is None:
        dim = O['dim_proj']

    layer_id = kwargs.pop('layer_id', 0)
    context_dim = kwargs.pop('context_dim', None)

    # embedding to gates transformation weights, biases
    params[_p(prefix, 'W', layer_id)] = np.concatenate([normal_weight(nin, dim), normal_weight(nin, dim)], axis=1)
    params[_p(prefix, 'b', layer_id)] = np.zeros((2 * dim,), dtype=fX)

    # recurrent transformation weights for gates
    params[_p(prefix, 'U', layer_id)] = np.concatenate([orthogonal_weight(dim),
                                                        orthogonal_weight(dim)], axis=1)

    # embedding to hidden state proposal weights, biases
    params[_p(prefix, 'Wx', layer_id)] = normal_weight(nin, dim)
    params[_p(prefix, 'bx', layer_id)] = np.zeros((dim,), dtype=fX)

    # recurrent transformation weights for hidden state proposal
    params[_p(prefix, 'Ux', layer_id)] = orthogonal_weight(dim)

    if context_dim is not None:
        params[_p(prefix, 'Wc', layer_id)] = np.concatenate([normal_weight(context_dim, dim),
                                                             normal_weight(context_dim, dim)], axis=1)
        params[_p(prefix, 'Wcx', layer_id)] = normal_weight(context_dim, dim)

    return params


def _gru_step_slice(
        src_mask, x_, xx_,
        ht_1,
        U, Ux):
    """GRU step function to be used by scan

    arguments (0) | sequences (3) | outputs-info (1) | non-seqs (2)

    ht_1: ([BS], [H])
    U: ([H], [H] + [H])
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


def _gru_step_slice_attention(
        mask, x_, xx_, context,
        ht_1,
        U, Ux, Wc, Wcx):
    """GRU step function with attention.
    
    context: ([BS], [Hc])
    Wc: ([Hc], [H] + [H])
    Wcx: ([Hc], [H])
    """

    _dim = Ux.shape[1]

    preact = T.nnet.sigmoid(T.dot(ht_1, U) + x_ + T.dot(context, Wc))

    # reset and update gates
    r = _slice(preact, 0, _dim)
    u = _slice(preact, 1, _dim)

    # hidden state proposal
    ht_tilde = T.tanh(T.dot(ht_1, Ux) * r + xx_ + T.dot(context, Wcx))

    # leaky integrate and obtain next hidden state
    ht = u * ht_1 + (1. - u) * ht_tilde
    ht = mask[:, None] * ht + (1. - mask)[:, None] * ht_1

    return ht


def gru_layer(P, state_below, O, prefix='gru', mask=None, **kwargs):
    """GRU layer
    
    input:
        state_below: ([Ts/t], [BS], x)     # x = [W] for src_embedding
        mask: ([Ts/t], [BS])
        context: ([Tt], [BS], [Hc])
    output: a list
        output[0]: hidden, ([Ts/t], [BS], [H])
    """

    layer_id = kwargs.pop('layer_id', 0)
    dropout_params = kwargs.pop('dropout_params', None)
    context = kwargs.pop('context', None)
    one_step = kwargs.pop('one_step', False)

    n_steps = state_below.shape[0]
    n_samples = state_below.shape[1] if state_below.ndim == 3 else 1
    dim = P[_p(prefix, 'Ux', layer_id)].shape[1]

    mask = T.alloc(1., n_steps, 1) if mask is None else mask

    # state_below is the input word embeddings
    # input to the gates, concatenated
    state_below_ = T.dot(state_below, P[_p(prefix, 'W', layer_id)]) + P[_p(prefix, 'b', layer_id)]
    # input to compute the hidden state proposal
    state_belowx = T.dot(state_below, P[_p(prefix, 'Wx', layer_id)]) + P[_p(prefix, 'bx', layer_id)]

    # prepare scan arguments
    init_states = [kwargs.pop('init_states', T.alloc(0., n_samples, dim))]

    if context is None:
        seqs = [mask, state_below_, state_belowx]
        shared_vars = [
            P[_p(prefix, 'U', layer_id)],
            P[_p(prefix, 'Ux', layer_id)],
        ]
        _step = _gru_step_slice
    else:
        seqs = [mask, state_below_, state_belowx, context]
        shared_vars = [
            P[_p(prefix, 'U', layer_id)],
            P[_p(prefix, 'Ux', layer_id)],
            P[_p(prefix, 'Wc', layer_id)],
            P[_p(prefix, 'Wcx', layer_id)],
        ]
        _step = _gru_step_slice_attention

    if one_step:
        outputs = _step(*(seqs + init_states + shared_vars))
    else:
        outputs, _ = theano.scan(
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
                        dim_nonlin=None, **kwargs):
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
    layer_id = kwargs.pop('layer_id', 0)

    W = np.concatenate([normal_weight(nin, dim),
                        normal_weight(nin, dim)], axis=1)
    params[_p(prefix, 'W', layer_id)] = W
    params[_p(prefix, 'b', layer_id)] = np.zeros((2 * dim,), dtype=fX)
    U = np.concatenate([orthogonal_weight(dim_nonlin),
                        orthogonal_weight(dim_nonlin)], axis=1)
    params[_p(prefix, 'U', layer_id)] = U

    Wx = normal_weight(nin_nonlin, dim_nonlin)
    params[_p(prefix, 'Wx', layer_id)] = Wx
    Ux = orthogonal_weight(dim_nonlin)
    params[_p(prefix, 'Ux', layer_id)] = Ux
    params[_p(prefix, 'bx', layer_id)] = np.zeros((dim_nonlin,), dtype=fX)

    U_nl = np.concatenate([orthogonal_weight(dim_nonlin),
                           orthogonal_weight(dim_nonlin)], axis=1)
    params[_p(prefix, 'U_nl', layer_id)] = U_nl
    params[_p(prefix, 'b_nl', layer_id)] = np.zeros((2 * dim_nonlin,), dtype=fX)

    Ux_nl = orthogonal_weight(dim_nonlin)
    params[_p(prefix, 'Ux_nl', layer_id)] = Ux_nl
    params[_p(prefix, 'bx_nl', layer_id)] = np.zeros((dim_nonlin,), dtype=fX)

    # context to LSTM
    Wc = normal_weight(dimctx, dim * 2)
    params[_p(prefix, 'Wc', layer_id)] = Wc

    Wcx = normal_weight(dimctx, dim)
    params[_p(prefix, 'Wcx', layer_id)] = Wcx

    # attention: combined -> hidden
    W_comb_att = normal_weight(dim, dimctx)
    params[_p(prefix, 'W_comb_att', layer_id)] = W_comb_att

    # attention: context -> hidden
    Wc_att = normal_weight(dimctx)
    params[_p(prefix, 'Wc_att', layer_id)] = Wc_att

    # attention: hidden bias
    b_att = np.zeros((dimctx,), dtype=fX)
    params[_p(prefix, 'b_att', layer_id)] = b_att

    # attention:
    U_att = normal_weight(dimctx, 1)
    params[_p(prefix, 'U_att', layer_id)] = U_att
    c_att = np.zeros((1,), dtype=fX)
    params[_p(prefix, 'c_tt', layer_id)] = c_att

    return params


def gru_cond_layer(P, state_below, O, prefix='gru', mask=None, context=None, one_step=False, init_memory=None,
                   init_state=None, context_mask=None, **kwargs):
    """Conditional GRU layer with Attention
    
    input:
        state_below: ([Tt], [BS], x)    # x = [W] for tgt_embedding
        mask: ([Tt], [BS])
        init_state: ([BS], [H])
        context: ([Tt], [BS], [Hc])
        context_mask: ([Tt], [BS])
    
    :return list of 3 outputs
        hidden_decoder: ([Tt], [BS], [H]), hidden states of the decoder gru
        context_decoder: ([Tt], [BS], [Hc]), weighted averages of context, generated by attention module
        alpha_decoder: ([Tt], [Bs], [Tt]), weights (alignment matrix)
    """

    layer_id = kwargs.pop('layer_id', 0)

    assert context, 'Context must be provided'
    assert context.ndim == 3, 'Context must be 3-d: #annotation * #sample * dim'
    if one_step:
        assert init_state, 'previous state must be provided'

    # Dimensions
    n_steps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1
    dim = P[_p(prefix, 'Wcx', layer_id)].shape[1]
    dropout_params = kwargs.pop('dropout_params', None)

    # Mask
    if mask is None:
        mask = T.alloc(1., n_steps, 1)

    # Initial/previous state
    if init_state is None:
        init_state = T.alloc(0., n_samples, dim)

    projected_context = T.dot(context, P[_p(prefix, 'Wc_att', layer_id)]) + P[_p(prefix, 'b_att', layer_id)]

    # projected x
    state_belowx = T.dot(state_below, P[_p(prefix, 'Wx', layer_id)]) + P[_p(prefix, 'bx', layer_id)]
    state_below_ = T.dot(state_below, P[_p(prefix, 'W', layer_id)]) + P[_p(prefix, 'b', layer_id)]

    def _step_slice(m_, x_, xx_,
                    h_, ctx_, alpha_,
                    projected_context_, context_,
                    U, Wc, W_comb_att, U_att, c_tt, Ux, Wcx, U_nl, Ux_nl, b_nl, bx_nl):
        preact1 = T.nnet.sigmoid(T.dot(h_, U) + x_)

        r1 = _slice(preact1, 0, dim)
        u1 = _slice(preact1, 1, dim)

        preactx1 = T.dot(h_, Ux) * r1 + xx_

        h1 = T.tanh(preactx1)

        h1 = u1 * h_ + (1. - u1) * h1
        h1 = m_[:, None] * h1 + (1. - m_)[:, None] * h_

        # attention
        pstate_ = T.dot(h1, W_comb_att)
        pctx__ = projected_context_ + pstate_[None, :, :]
        # pctx__ += xc_
        pctx__ = T.tanh(pctx__)

        alpha = T.dot(pctx__, U_att) + c_tt
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
        alpha = T.exp(alpha)
        if context_mask:
            alpha = alpha * context_mask
        alpha = alpha / alpha.sum(0, keepdims=True)
        ctx_ = (context_ * alpha[:, :, None]).sum(0)  # current context

        # GRU 2 (with attention)
        preact2 = T.nnet.sigmoid(T.dot(h1, U_nl) + b_nl + T.dot(ctx_, Wc))

        r2 = _slice(preact2, 0, dim)
        u2 = _slice(preact2, 1, dim)

        preactx2 = (T.dot(h1, Ux_nl) + bx_nl) * r2 + T.dot(ctx_, Wcx)

        h2 = T.tanh(preactx2)

        h2 = u2 * h1 + (1. - u2) * h2
        h2 = m_[:, None] * h2 + (1. - m_)[:, None] * h1

        return h2, ctx_, alpha.T

    seqs = [mask, state_below_, state_belowx]
    _step = _step_slice

    shared_vars = [P[_p(prefix, 'U', layer_id)],
                   P[_p(prefix, 'Wc', layer_id)],
                   P[_p(prefix, 'W_comb_att', layer_id)],
                   P[_p(prefix, 'U_att', layer_id)],
                   P[_p(prefix, 'c_tt', layer_id)],
                   P[_p(prefix, 'Ux', layer_id)],
                   P[_p(prefix, 'Wcx', layer_id)],
                   P[_p(prefix, 'U_nl', layer_id)],
                   P[_p(prefix, 'Ux_nl', layer_id)],
                   P[_p(prefix, 'b_nl', layer_id)],
                   P[_p(prefix, 'bx_nl', layer_id)]]

    if one_step:
        result = _step(*(seqs + [init_state, None, None, projected_context, context] + shared_vars))
    else:
        result, _ = theano.scan(
            _step,
            sequences=seqs,
            outputs_info=[init_state,
                          T.alloc(0., n_samples,
                                  context.shape[2]),
                          T.alloc(0., n_samples,
                                  context.shape[0])],
            non_sequences=[projected_context, context] + shared_vars,
            name=_p(prefix, '_layers'),
            n_steps=n_steps,
            profile=profile,
            strict=True,
        )

    if dropout_params:
        result = list(result)
        result[0] = dropout_layer(result[0], *dropout_params)

    return result


def param_init_lstm(O, params, prefix='lstm', nin=None, dim=None, **kwargs):
    if nin is None:
        nin = O['dim_proj']
    if dim is None:
        dim = O['dim_proj']

    layer_id = kwargs.pop('layer_id', 0)
    context_dim = kwargs.pop('context_dim', None)

    params[_p(prefix, 'W', layer_id)] = np.concatenate([
        normal_weight(nin, dim),
        normal_weight(nin, dim),
        normal_weight(nin, dim),
        normal_weight(nin, dim),
    ], axis=1)

    params[_p(prefix, 'U', layer_id)] = np.concatenate([
        orthogonal_weight(dim),
        orthogonal_weight(dim),
        orthogonal_weight(dim),
        orthogonal_weight(dim),
    ], axis=1)

    params[_p(prefix, 'b', layer_id)] = np.zeros((4 * dim,), dtype=fX)

    if context_dim is not None:
        # todo Add context
        pass

    return params


def _lstm_step_slice(
        mask_, x_,
        h_, c_,
        U):
    _dim = U.shape[1] // 4

    preact = T.dot(h_, U) + x_

    i = T.nnet.sigmoid(_slice(preact, 0, _dim))
    f = T.nnet.sigmoid(_slice(preact, 1, _dim))
    o = T.nnet.sigmoid(_slice(preact, 2, _dim))
    c = T.tanh(_slice(preact, 3, _dim))

    c = f * c_ + i * c
    c = mask_[:, None] * c + (1. - mask_)[:, None] * c_

    h = o * T.tanh(c)
    h = mask_[:, None] * h + (1. - mask_)[:, None] * h_

    return h, c


def _lstm_step_slice_attention(
        mask_, x_, context,
        h_, c_,
        U, Wc):
    _dim = U.shape[1] // 4

    preact = T.dot(h_, U) + x_ + T.dot(context, Wc)

    i = T.nnet.sigmoid(_slice(preact, 0, _dim))
    f = T.nnet.sigmoid(_slice(preact, 1, _dim))
    o = T.nnet.sigmoid(_slice(preact, 2, _dim))
    c = T.tanh(_slice(preact, 3, _dim))

    c = f * c_ + i * c
    c = mask_[:, None] * c + (1. - mask_)[:, None] * c_

    h = o * T.tanh(c)
    h = mask_[:, None] * h + (1. - mask_)[:, None] * h_

    return h, c


def lstm_layer(P, state_below, O, prefix='lstm', mask=None, **kwargs):
    """LSTM layer
    
    inputs and outputs are same as GRU layer.
    """

    layer_id = kwargs.pop('layer_id', 0)
    dropout_params = kwargs.pop('dropout_params', None)
    context = kwargs.pop('context', None)
    one_step = kwargs.pop('one_step', False)

    n_steps = state_below.shape[0]
    n_samples = state_below.shape[1] if state_below.ndim == 3 else 1
    dim = P[_p(prefix, 'U', layer_id)].shape[1] // 4

    mask = T.alloc(1., n_steps, 1) if mask is None else mask

    state_below = T.dot(state_below, P[_p(prefix, 'W', layer_id)]) + P[_p(prefix, 'b', layer_id)]

    # prepare scan arguments
    init_states = [kwargs.pop('init_states', T.alloc(0., n_samples, dim)),
                   T.alloc(0., n_samples, dim)]

    if context is None:
        seqs = [mask, state_below]
        shared_vars = [P[_p(prefix, 'U', layer_id)]]
        _step = _lstm_step_slice
    else:
        seqs = [mask, state_below, context]
        shared_vars = [P[_p(prefix, 'U', layer_id)],
                       P[_p(prefix, 'Wc', layer_id)]]
        _step = _lstm_step_slice_attention

    if one_step:
        outputs = _step(*(seqs + init_states + shared_vars))
    else:
        outputs, _ = theano.scan(
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


def lstm_cond_layer(P, state_below, O, prefix='lstm', mask=None, context=None, one_step=False, init_memory=None,
                    init_state=None, context_mask=None, **kwargs):
    """Conditional LSTM layer with attention
    
    inputs and outputs are same as GRU cond layer.
    """

    layer_id = kwargs.pop('layer_id', 0)

    assert context, 'Context must be provided'
    assert context.ndim == 3, 'Context must be 3-d: #annotation * #sample * dim'
    if one_step:
        assert init_state, 'previous state must be provided'

    # Dimensions
    n_steps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1
    dim = P[_p(prefix, 'Wcx', layer_id)].shape[1]
    dropout_params = kwargs.pop('dropout_params', None)

    # Mask
    if mask is None:
        mask = T.alloc(1., n_steps, 1)

    # Initial/previous state
    if init_state is None:
        init_state = T.alloc(0., n_samples, dim)

    projected_context = T.dot(context, P[_p(prefix, 'Wc_att', layer_id)]) + P[_p(prefix, 'b_att', layer_id)]

    # Projected x
    state_below = T.dot(state_below, P[_p(prefix, 'W', layer_id)]) + P[_p(prefix, 'b', layer_id)]


# todo: implement residual connection


# layers: 'name': ('parameter initializer', 'builder')
layers = {
    'ff': (param_init_feed_forward, feed_forward),
    'gru': (param_init_gru, gru_layer),
    'gru_cond': (param_init_gru_cond, gru_cond_layer),
    'lstm': (param_init_lstm, lstm_layer),
}


def get_layer(name):
    fns = layers[name]
    return fns[0], fns[1]


def get_init(name):
    return layers[name][0]


def get_build(name):
    return layers[name][1]


__all__ = [
    'tanh',
    'linear',
    'attention_layer',
    'param_init_feed_forward',
    'feed_forward',
    'param_init_gru',
    'gru_layer',
    'param_init_gru_cond',
    'gru_cond_layer',
    'layers',
    'get_layer',
    'get_build',
    'get_init',
]
