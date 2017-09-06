#! /usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import theano
from theano import tensor as T

from ..constants import fX, profile
from .basic import _slice, _attention, dropout_layer, layer_normalization_layer, _attention_trg
from ..utility.utils import _p, normal_weight, orthogonal_weight, concatenate

__author__ = 'fyabc'


def param_init_lstm(O, params, prefix='lstm', nin=None, dim=None, **kwargs):
    if nin is None:
        nin = O['dim_proj']
    if dim is None:
        dim = O['dim_proj']

    layer_id = kwargs.pop('layer_id', 0)
    context_dim = kwargs.pop('context_dim', None)
    multi = 'multi' in O.get('unit', 'lstm')
    unit_size = kwargs.pop('unit_size', O.get('unit_size', 2))
    use_layer_normalization = O.get('use_LN', False)

    if not multi:
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
            params[_p(prefix, 'Wc', layer_id)] = np.concatenate([
                normal_weight(context_dim, dim),
                normal_weight(context_dim, dim),
                normal_weight(context_dim, dim),
                normal_weight(context_dim, dim),
            ], axis=1)
    else:
        params[_p(prefix, 'W', layer_id)] = np.stack([np.concatenate([
            normal_weight(nin, dim),
            normal_weight(nin, dim),
            normal_weight(nin, dim),
            normal_weight(nin, dim),
        ], axis=1) for _ in xrange(unit_size)], axis=0)

        params[_p(prefix, 'U', layer_id)] = np.stack([np.concatenate([
            orthogonal_weight(dim),
            orthogonal_weight(dim),
            orthogonal_weight(dim),
            orthogonal_weight(dim),
        ], axis=1) for _ in xrange(unit_size)], axis=0)

        params[_p(prefix, 'b', layer_id)] = np.zeros((unit_size, 4 * dim,), dtype=fX)

        if context_dim is not None:
            params[_p(prefix, 'Wc', layer_id)] = np.stack([np.concatenate([
                normal_weight(context_dim, dim),
                normal_weight(context_dim, dim),
                normal_weight(context_dim, dim),
                normal_weight(context_dim, dim),
            ], axis=1) for _ in xrange(unit_size)], axis=0)

    if use_layer_normalization:
        params[_p(prefix, 'alpha_h', layer_id)] = np.ones((4*dim,),dtype='float32')
        params[_p(prefix, 'beta_h', layer_id)] = np.zeros((4*dim,), dtype='float32')
        params[_p(prefix, 'alpha_x', layer_id)] = np.ones((4*dim,),dtype='float32')
        params[_p(prefix, 'beta_x', layer_id)] = np.zeros((4*dim,), dtype='float32')
        params[_p(prefix, 'alpha_c', layer_id)] = np.ones((dim,), dtype='float32')
        params[_p(prefix, 'beta_c', layer_id)] = np.zeros((dim,), dtype='float32')
        if context_dim is not None:
            params[_p(prefix, 'alpha_ctx', layer_id)] = np.ones((4*dim,),dtype='float32')
            params[_p(prefix, 'beta_ctx', layer_id)] = np.zeros((4*dim,), dtype='float32')

    return params

def _lstm_step_kernel(preact, mask_, h_, c_, _dim):
    i = T.nnet.sigmoid(_slice(preact, 0, _dim))
    f = T.nnet.sigmoid(_slice(preact, 1, _dim))
    o = T.nnet.sigmoid(_slice(preact, 2, _dim))
    c = T.tanh(_slice(preact, 3, _dim))

    c = f * c_ + i * c
    c = mask_[:, None] * c + (1. - mask_)[:, None] * c_

    h = o * T.tanh(c)
    h = mask_[:, None] * h + (1. - mask_)[:, None] * h_

    return i, f, o, c, h


def _ln_lstm_step_kernel(preact, mask_, h_, c_, _dim, alpha_, beta_):
    i = T.nnet.sigmoid(_slice(preact, 0, _dim))
    f = T.nnet.sigmoid(_slice(preact, 1, _dim))
    o = T.nnet.sigmoid(_slice(preact, 2, _dim))
    c = T.tanh(_slice(preact, 3, _dim))

    c = f * c_ + i * c
    c = mask_[:, None] * c + (1. - mask_)[:, None] * c_

    h = o * T.tanh(layer_normalization_layer(c, alpha_, beta_))
    h = mask_[:, None] * h + (1. - mask_)[:, None] * h_

    return i, f, o, c, h

def _lstm_step_slice(
        mask_, x_,
        h_, c_,
        U):
    _dim = U.shape[1] // 4
    preact = T.dot(h_, U) + x_

    i, f, o, c, h = _lstm_step_kernel(preact, mask_, h_, c_, _dim)
    return h, c


def _ln_lstm_step_slice(
        mask_, x_,
        h_, c_,
        U, alpha_h, beta_h, alpha_x, beta_x, alpha_c, beta_c, input_bias):
    _dim = U.shape[1] // 4

    zh = layer_normalization_layer(T.dot(h_, U), alpha_h, beta_h)
    zx = layer_normalization_layer(x_, alpha_x, beta_x)
    i, f, o, c, h = _ln_lstm_step_kernel(zh + zx + input_bias, mask_, h_, c_, _dim, alpha_c, beta_c)
    return h, c

def _lstm_step_slice_gates(
        mask_, x_,
        h_, c_, i_, f_, o_,
        U):
    _dim = U.shape[1] // 4
    preact = T.dot(h_, U) + x_

    i, f, o, c, h = _lstm_step_kernel(preact, mask_, h_, c_, _dim)
    return h, c, i, f, o


def _lstm_step_slice_attention(
        mask_, x_, context,
        h_, c_,
        U, Wc):
    _dim = U.shape[1] // 4
    preact = T.dot(h_, U) + x_ + T.dot(context, Wc)

    i, f, o, c, h = _lstm_step_kernel(preact, mask_, h_, c_, _dim)
    return h, c

def _ln_lstm_step_slice_attention(
        mask_, x_, context,
        h_, c_,
        U, Wc, alpha_h, beta_h, alpha_x, beta_x, alpha_ctx, beta_ctx, alpha_c, beta_c, input_bias):
    _dim = U.shape[1] // 4
    preact = layer_normalization_layer(T.dot(h_, U), alpha_h, beta_h) + \
             layer_normalization_layer(x_, alpha_x, beta_x) + \
             layer_normalization_layer(T.dot(context, Wc), alpha_ctx, beta_ctx) + input_bias

    i, f, o, c, h = _ln_lstm_step_kernel(preact, mask_, h_, c_, _dim, alpha_c, beta_c)
    return h, c

def _lstm_step_slice_attention_gates(
        mask_, x_, context,
        h_, c_, i_, f_, o_,
        U, Wc):
    _dim = U.shape[1] // 4
    preact = T.dot(h_, U) + x_ + T.dot(context, Wc)

    i, f, o, c, h = _lstm_step_kernel(preact, mask_, h_, c_, _dim)
    return h, c, i, f, o


def lstm_layer(P, state_below, O, prefix='lstm', mask=None, **kwargs):
    """LSTM layer

    inputs and outputs are same as GRU layer.

    outputs[1]: hidden memory
    """

    layer_id = kwargs.pop('layer_id', 0)
    dropout_params = kwargs.pop('dropout_params', None)
    context = kwargs.pop('context', None)
    one_step = kwargs.pop('one_step', False)
    init_state = kwargs.pop('init_state', None)
    init_memory = kwargs.pop('init_memory', None)
    multi = 'multi' in O.get('unit', 'lstm')
    unit_size = kwargs.pop('unit_size', O.get('unit_size', 2))
    # FIXME: multi-gru/lstm do NOT support get_gates now
    get_gates = kwargs.pop('get_gates', False)
    use_LN = kwargs.pop('use_LN', False)

    kw_ret = {}

    n_steps = state_below.shape[0] if state_below.ndim == 3 else 1
    n_samples = state_below.shape[1] if state_below.ndim == 3 else state_below.shape[0]
    if multi:
        dim = P[_p(prefix, 'U', layer_id)][0].shape[1] // 4
    else:
        dim = P[_p(prefix, 'U', layer_id)].shape[1] // 4

    mask = T.alloc(1., n_steps, 1) if mask is None else mask

    if multi:
        state_below = T.concatenate([
            T.dot(state_below, P[_p(prefix, 'W', layer_id)][j]) + P[_p(prefix, 'b', layer_id)][j]
            for j in range(unit_size)
        ], axis=-1)
    else:
        if use_LN:
            # Warning: the P[_p(prefix, 'b', layer_id)] would be used as input_bias in lstm cell
            state_below = T.dot(state_below, P[_p(prefix, 'W', layer_id)])
        else:
            state_below = T.dot(state_below, P[_p(prefix, 'W', layer_id)]) + P[_p(prefix, 'b', layer_id)]

    def _step_slice(mask_, x_, h_, c_, U):
        h_tmp = h_
        c_tmp = c_
        for j in range(unit_size):
            x = _slice(x_, j, 4 * dim)
            h, c = _lstm_step_slice(mask_, x, h_tmp, c_tmp, U[j])
            h_tmp = h
            c_tmp = c
        return h, c

    def _step_slice_attention(mask_, x_, context, h_, c_, U, Wc):
        h_tmp = h_
        c_tmp = c_
        for j in range(unit_size):
            x = _slice(x_, j, 4 * dim)
            h, c = _lstm_step_slice_attention(mask_, x, context, h_tmp, c_tmp, U[j], Wc[j])
            h_tmp = h
            c_tmp = c
        return h, c

    # prepare scan arguments
    init_states = [T.alloc(0., n_samples, dim) if init_state is None else init_state,
                   T.alloc(0., n_samples, dim) if init_memory is None else init_memory, ]
    if get_gates:
        init_states.extend([T.alloc(0., n_samples, dim) for _ in range(3)])

    if context is None:
        seqs = [mask, state_below]
        shared_vars = [P[_p(prefix, 'U', layer_id)]]
        if use_LN:
            shared_vars += [
                    P[_p(prefix, 'alpha_h', layer_id)],
                    P[_p(prefix, 'beta_h', layer_id)],
                    P[_p(prefix, 'alpha_x', layer_id)],
                    P[_p(prefix, 'beta_x', layer_id)],
                    P[_p(prefix, 'alpha_c', layer_id)],
                    P[_p(prefix, 'beta_c', layer_id)],
                    P[_p(prefix, 'b', layer_id)]
                ]
        if multi:
            _step = _step_slice
        else:
            if use_LN:
                _step = _ln_lstm_step_slice
            elif get_gates:
                _step = _lstm_step_slice_gates
            else:
                _step = _lstm_step_slice
    else:
        seqs = [mask, state_below, context]
        shared_vars = [P[_p(prefix, 'U', layer_id)],
                       P[_p(prefix, 'Wc', layer_id)]]
        if use_LN:
            shared_vars += [
                P[_p(prefix, 'alpha_h', layer_id)],
                P[_p(prefix, 'beta_h', layer_id)],
                P[_p(prefix, 'alpha_x', layer_id)],
                P[_p(prefix, 'beta_x', layer_id)],
                P[_p(prefix, 'alpha_ctx', layer_id)],
                P[_p(prefix, 'beta_ctx', layer_id)],
                P[_p(prefix, 'alpha_c', layer_id)],
                P[_p(prefix, 'beta_c', layer_id)],
                P[_p(prefix, 'b', layer_id)]
            ]
            _step =_ln_lstm_step_slice_attention
        elif multi:
            _step = _step_slice_attention
        else:
            if get_gates:
                _step = _lstm_step_slice_attention_gates
            else:
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

    kw_ret['hidden_without_dropout'] = outputs[0]
    kw_ret['memory_output'] = outputs[1]

    if get_gates:
        kw_ret['input_gates'] = outputs[2]
        kw_ret['forget_gates'] = outputs[3]
        kw_ret['output_gates'] = outputs[4]

    outputs = [outputs[0], outputs[1], kw_ret]

    if dropout_params:
        outputs[0] = dropout_layer(outputs[0], *dropout_params)

    return outputs


def param_init_lstm_cond(O, params, prefix='lstm_cond', nin=None, dim=None, dimctx=None, nin_nonlin=None,
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
    multi = 'multi' in O.get('unit', 'lstm_cond')
    unit_size = kwargs.pop('unit_size', O.get('cond_unit_size', 2))
    use_layer_normalization = O.get('use_LN', False)
    dense_attention = O['densely_connected'] and O['dense_attention']

    if not multi:
        params[_p(prefix, 'W', layer_id)] = np.concatenate([
            normal_weight(nin, dim),
            normal_weight(nin, dim),
            normal_weight(nin, dim),
            normal_weight(nin_nonlin, dim),
        ], axis=1)
        params[_p(prefix, 'b', layer_id)] = np.zeros((4 * dim,), dtype=fX)
        params[_p(prefix, 'U', layer_id)] = np.concatenate([
            orthogonal_weight(dim_nonlin),
            orthogonal_weight(dim_nonlin),
            orthogonal_weight(dim_nonlin),
            orthogonal_weight(dim_nonlin),
        ], axis=1)

        params[_p(prefix, 'U_nl', layer_id)] = np.concatenate([
            orthogonal_weight(dim_nonlin),
            orthogonal_weight(dim_nonlin),
            orthogonal_weight(dim_nonlin),
            orthogonal_weight(dim_nonlin),
        ], axis=1)
        params[_p(prefix, 'b_nl', layer_id)] = np.zeros((4 * dim_nonlin,), dtype=fX)

        # context to LSTM
        params[_p(prefix, 'Wc', layer_id)] = normal_weight(dimctx, dim * 4)
    else:
        params[_p(prefix, 'W', layer_id)] = np.stack([np.concatenate([
            normal_weight(nin, dim),
            normal_weight(nin, dim),
            normal_weight(nin, dim),
            normal_weight(nin_nonlin, dim),
        ], axis=1) for _ in xrange(unit_size)], axis=0)
        params[_p(prefix, 'b', layer_id)] = np.zeros((unit_size, 4 * dim,), dtype=fX)
        params[_p(prefix, 'U', layer_id)] = np.stack([np.concatenate([
            orthogonal_weight(dim_nonlin),
            orthogonal_weight(dim_nonlin),
            orthogonal_weight(dim_nonlin),
            orthogonal_weight(dim_nonlin),
        ], axis=1) for _ in xrange(unit_size)], axis=0)

        params[_p(prefix, 'U_nl', layer_id)] = np.stack([np.concatenate([
            orthogonal_weight(dim_nonlin),
            orthogonal_weight(dim_nonlin),
            orthogonal_weight(dim_nonlin),
            orthogonal_weight(dim_nonlin),
        ], axis=1) for _ in xrange(unit_size)], axis=0)
        params[_p(prefix, 'b_nl', layer_id)] = np.zeros((unit_size, 4 * dim_nonlin,), dtype=fX)

        # context to LSTM
        params[_p(prefix, 'Wc', layer_id)] = np.stack([
            normal_weight(dimctx, dim * 4) for _ in xrange(unit_size)], axis=0)

    if dense_attention:
        dim_word = O['dim_word']
        for i in xrange(O['n_encoder_layers'] + 1):
            if i == 0:
                params[_p(prefix, 'W_comb_att', layer_id, i)] = normal_weight(dim, 2 * dim_word)
                params[_p(prefix, 'Wc_att', layer_id, i)] = normal_weight(2 * dim_word)
                params[_p(prefix, 'b_att', layer_id, i)] = np.zeros((2 * dim_word,), dtype=fX)
                params[_p(prefix, 'U_att', layer_id, i)] = normal_weight(2* dim_word, 1)
            else:
                params[_p(prefix, 'W_comb_att', layer_id, i)] = normal_weight(dim, 2 * dim)
                params[_p(prefix, 'Wc_att', layer_id, i)] = normal_weight(2 * dim)
                params[_p(prefix, 'b_att', layer_id, i)] = np.zeros((2 * dim,), dtype=fX)
                params[_p(prefix, 'U_att', layer_id, i)] = normal_weight(2 * dim, 1)
            params[_p(prefix, 'c_tt', layer_id, i)] = np.zeros((1,), dtype=fX)
    else:
        # attention: combined -> hidden
        params[_p(prefix, 'W_comb_att', layer_id)] = normal_weight(dim, dimctx)
        # attention: context -> hidden
        params[_p(prefix, 'Wc_att', layer_id)] = normal_weight(dimctx)
        # attention: hidden bias
        params[_p(prefix, 'b_att', layer_id)] = np.zeros((dimctx,), dtype=fX)
        # attention:
        params[_p(prefix, 'U_att', layer_id)] = normal_weight(dimctx, 1)
        params[_p(prefix, 'c_tt', layer_id)] = np.zeros((1,), dtype=fX)

    if use_layer_normalization:
        params[_p(prefix, 'alpha_h', layer_id)] = np.ones((4 * dim,), dtype='float32')
        params[_p(prefix, 'beta_h', layer_id)] = np.zeros((4 * dim,), dtype='float32')
        params[_p(prefix, 'alpha_x', layer_id)] = np.ones((4 * dim,), dtype='float32')
        params[_p(prefix, 'beta_x', layer_id)] = np.zeros((4 * dim,), dtype='float32')
        params[_p(prefix, 'alpha_c', layer_id)] = np.ones((dim,), dtype='float32')
        params[_p(prefix, 'beta_c', layer_id)] = np.zeros((dim,), dtype='float32')

        params[_p(prefix, 'alpha_h2', layer_id)] = np.ones((4 * dim,), dtype='float32')
        params[_p(prefix, 'beta_h2', layer_id)] = np.zeros((4 * dim,), dtype='float32')
        params[_p(prefix, 'alpha_ctx', layer_id)] = np.ones((4 * dim,), dtype='float32')
        params[_p(prefix, 'beta_ctx', layer_id)] = np.zeros((4 * dim,), dtype='float32')
        params[_p(prefix, 'alpha_c2', layer_id)] = np.ones((dim,), dtype='float32')
        params[_p(prefix, 'beta_c2', layer_id)] = np.zeros((dim,), dtype='float32')

    return params


def lstm_cond_layer(P, state_below, O, prefix='lstm', mask=None, context=None, one_step=False, init_memory=None,
                    init_state=None, context_mask=None, **kwargs):
    """Conditional LSTM layer with attention

    inputs and outputs are same as GRU cond layer.
    """

    layer_id = kwargs.pop('layer_id', 0)
    multi = 'multi' in O.get('unit', 'lstm_cond')
    unit_size = kwargs.pop('unit_size', O.get('cond_unit_size', 2))
    # FIXME: multi-gru/lstm do NOT support get_gates now
    get_gates = kwargs.pop('get_gates', False)
    use_LN = kwargs.pop('use_LN', False)
    dense_attention = O['densely_connected'] and O['dense_attention']

    kw_ret = {}

    assert context, 'Context must be provided'
    assert context.ndim == 3, 'Context must be 3-d: #annotation * #sample * dim'
    if one_step:
        assert init_state, 'previous state must be provided'

    # Dimensions
    n_steps = state_below.shape[0] if state_below.ndim == 3 else 1
    n_samples = state_below.shape[1] if state_below.ndim == 3 else state_below.shape[0]
    if multi:
        dim = P[_p(prefix, 'Wc', layer_id)][0].shape[1] // 4
    else:
        dim = P[_p(prefix, 'Wc', layer_id)].shape[1] // 4
    dropout_params = kwargs.pop('dropout_params', None)

    # Mask
    if mask is None:
        mask = T.alloc(1., n_steps, 1)

    # Initial/previous state
    if init_state is None:
        init_state = T.alloc(0., n_samples, dim)

    if dense_attention：
    dim_word = O['dim_word']
    dim = self.O['dim']
    for i in xrange(O['n_encoder_layers'] + 1):
        if i == 0:
            projected_context = T.dot(context[:, :, :2 * dim_word], P[_p(prefix, 'Wc_att', layer_id, i)]) + P[_p(prefix, 'b_att', layer_id, i)]
        else:
            projected_context = concatenate([projected_context, T.dot(context[:, :, 2*(dim_word+(i-1)*dim):2*(dim_word+i*dim)], P[_p(prefix, 'Wc_att', layer_id, i)]) + P[_p(prefix, 'b_att', layer_id, i)]], axis=projected_context.ndim - 1)
    else:
        projected_context = T.dot(context, P[_p(prefix, 'Wc_att', layer_id)]) + P[_p(prefix, 'b_att', layer_id)]

    # Projected x
    if use_LN:
        state_below = T.dot(state_below, P[_p(prefix, 'W', layer_id)])
        # Warning:  still, ``P[_p(prefix, 'b', layer_id)]'' will be thrown to layer normalization parts
    else:
        if multi:
            state_below = T.concatenate([
                T.dot(state_below, P[_p(prefix, 'W', layer_id)][j]) + P[_p(prefix, 'b', layer_id)][j]
                for j in range(unit_size)
            ], axis=-1)
        else:
            state_below = T.dot(state_below, P[_p(prefix, 'W', layer_id)]) + P[_p(prefix, 'b', layer_id)]

    def _one_step_attention_slice(mask_, h1, c1, ctx_, Wc, U_nl, b_nl):
        preact2 = T.dot(h1, U_nl) + b_nl + T.dot(ctx_, Wc)

        i2 = T.nnet.sigmoid(_slice(preact2, 0, dim))
        f2 = T.nnet.sigmoid(_slice(preact2, 1, dim))
        o2 = T.nnet.sigmoid(_slice(preact2, 2, dim))
        c2 = T.tanh(_slice(preact2, 3, dim))

        c2 = f2 * c1 + i2 * c2
        c2 = mask_[:, None] * c2 + (1. - mask_)[:, None] * c1

        h2 = o2 * T.tanh(c2)
        h2 = mask_[:, None] * h2 + (1. - mask_)[:, None] * h1

        if get_gates:
            return h2, c2, i2, f2, o2

        return h2, c2

    def _ln_attention_slice(mask_, h1, c1, ctx_, Wc, U_nl, b_nl,
                                  alpha_h, beta_h, alpha_ctx, beta_ctx, alpha_c, beta_c):
        preact2 = layer_normalization_layer(T.dot(h1, U_nl), alpha_h, beta_h) + b_nl + \
                  layer_normalization_layer(T.dot(ctx_, Wc), alpha_ctx, beta_ctx)

        i2 = T.nnet.sigmoid(_slice(preact2, 0, dim))
        f2 = T.nnet.sigmoid(_slice(preact2, 1, dim))
        o2 = T.nnet.sigmoid(_slice(preact2, 2, dim))
        c2 = T.tanh(_slice(preact2, 3, dim))

        c2 = f2 * c1 + i2 * c2
        c2 = mask_[:, None] * c2 + (1. - mask_)[:, None] * c1

        h2 = o2 * T.tanh(layer_normalization_layer(c2, alpha_c, beta_c))
        h2 = mask_[:, None] * h2 + (1. - mask_)[:, None] * h1

        if get_gates:
            return h2, c2, i2, f2, o2

        return h2, c2

    def _step_slice(mask_, x_,
                    h_, c_, ctx_, alpha_,
                    projected_context_, context_,
                    U, Wc, U_nl, b_nl, *args):
        # LSTM 1
        h1, c1 = _lstm_step_slice(mask_, x_, h_, c_, U)

        # Attention
        ctx_, alpha = _attention(h1, projected_context_, context_, context_mask=context_mask, dense_attention=dense_attention, dim_word=O['dim_word'], dim=O['dim'], n_enc=O['n_encoder_layers'], *args)

        # LSTM 2 (with attention)
        h2, c2 = _one_step_attention_slice(mask_, h1, c1, ctx_, Wc, U_nl, b_nl)

        return h2, c2, ctx_, alpha.T

    def _ln_step_slice(mask_, x_,
                       h_, c_, ctx_, alpha_,
                       projected_context_, context_,
                       U, Wc, U_nl, b_nl,
                       alpha_h, beta_h, alpha_x, beta_x, alpha_c, beta_c, input_bias,
                       alpha_h2, beta_h2, alpha_ctx, beta_ctx, alpha_c2, beta_c2, *args):
        # LSTM 1
        h1, c1 = _ln_lstm_step_slice(mask_, x_, h_, c_, U,
                                     alpha_h, beta_h, alpha_x, beta_x, alpha_c, beta_c, input_bias)

        # Attention
        ctx_, alpha = _attention(h1, projected_context_, context_, W_comb_att, U_att, c_tt, context_mask=context_mask, dense_attention=dense_attention, dim_word=O['dim_word'], dim=O['dim'], n_enc=O['n_encoder_layers'], *args)

        # LSTM 2 (with attention)
        h2, c2 = _ln_attention_slice(mask_, h1, c1, ctx_, Wc, U_nl, b_nl,
                                           alpha_h2, beta_h2, alpha_ctx, beta_ctx, alpha_c2, beta_c2)

        return h2, c2, ctx_, alpha.T

    # todo: implement it
    def _step_slice_gates(mask_, x_,
                          h_, c_, ctx_, alpha_, i1_, f1_, o1_, i2_, f2_, o2_,
                          projected_context_, context_,
                          U, Wc, U_nl, b_nl, *args):
        # LSTM 1
        h1, c1, i1, f1, o1 = _lstm_step_slice_gates(mask_, x_, h_, c_, i1_, f1_, o1_, U)

        # Attention
        ctx_, alpha = _attention(h1, projected_context_, context_, W_comb_att, U_att, c_tt, context_mask=context_mask, dense_attention=dense_attention, dim_word=O['dim_word'], dim=O['dim'], n_enc=O['n_encoder_layers'], *args)

        # LSTM 2 (with attention)
        h2, c2, i2, f2, o2 = _one_step_attention_slice(mask_, h1, c1, ctx_, Wc, U_nl, b_nl)

        return h2, c2, ctx_, alpha.T, i1, f1, o1, i2, f2, o2

    def _multi_step_slice(mask_, x_,
                          h_, c_, ctx_, alpha_,
                          projected_context_, context_,
                          U, Wc, U_nl, b_nl, *args):
        # LSTM 1
        h_tmp = h_
        c_tmp = c_
        for j in range(unit_size):
            x = _slice(x_, j, 4 * dim)
            h1, c1 = _lstm_step_slice(mask_, x, h_tmp, c_tmp, U[j])
            h_tmp = h1
            c_tmp = c1

        # Attention
        ctx_, alpha = _attention(h1, projected_context_, context_, W_comb_att, U_att, c_tt, context_mask=context_mask, dense_attention=dense_attention, dim_word=O['dim_word'], dim=O['dim'], n_enc=O['n_encoder_layers'], *args)

        # LSTM 2 (with attention)
        h_tmp_att = h1
        c_tmp_att = c1
        for j in range(unit_size):
            h2, c2 = _one_step_attention_slice(mask_, h_tmp_att, c_tmp_att, ctx_, Wc[j], U_nl[j], b_nl[j])
            h_tmp_att = h2
            c_tmp_att = c2

        return h2, c2, ctx_, alpha.T

    # Prepare scan arguments
    seqs = [mask, state_below]
    if use_LN:
        _step = _ln_step_slice
    elif multi:
        _step = _multi_step_slice
    else:
        if get_gates:
            _step = _step_slice_gates
        else:
            _step = _step_slice
    init_states = [
        init_state,
        T.alloc(0., n_samples, dim) if init_memory is None else init_memory,
        T.alloc(0., n_samples, context.shape[2]),
        T.alloc(0., n_samples, context.shape[0]),
    ]
    if get_gates:
        init_states.extend([T.alloc(0., n_samples, dim) for _ in range(6)])

    shared_vars = [
        P[_p(prefix, 'U', layer_id)],
        P[_p(prefix, 'Wc', layer_id)],
        
        P[_p(prefix, 'U_nl', layer_id)],
        P[_p(prefix, 'b_nl', layer_id)],
    ]

    if use_LN:
        shared_vars += [
            P[_p(prefix, 'alpha_h', layer_id)],
            P[_p(prefix, 'beta_h', layer_id)],
            P[_p(prefix, 'alpha_x', layer_id)],
            P[_p(prefix, 'beta_x', layer_id)],
            P[_p(prefix, 'alpha_c', layer_id)],
            P[_p(prefix, 'beta_c', layer_id)],
            P[_p(prefix, 'b', layer_id)],
            P[_p(prefix, 'alpha_h2', layer_id)],
            P[_p(prefix, 'beta_h2', layer_id)],
            P[_p(prefix, 'alpha_ctx', layer_id)],
            P[_p(prefix, 'beta_ctx', layer_id)],
            P[_p(prefix, 'alpha_c2', layer_id)],
            P[_p(prefix, 'beta_c2', layer_id)],
        ]

    if dense_attention:
        shared_vars += [
            P[_p(prefix, 'W_comb_att', layer_id, i)] for i in xrange(O['n_encoder_layers'] + 1) 
        ]
        shared_vars += [
            P[_p(prefix, 'U_att', layer_id, i)] for i in xrange(O['n_encoder_layers'] + 1) 
        ]
        shared_vars += [
            P[_p(prefix, 'c_tt', layer_id, i)] for i in xrange(O['n_encoder_layers'] + 1) 
        ]

    else:
        shared_vars += [
            P[_p(prefix, 'W_comb_att', layer_id)],
            P[_p(prefix, 'U_att', layer_id)],
            P[_p(prefix, 'c_tt', layer_id)],
        ]

    if one_step:
        result = _step(*(seqs + init_states + [projected_context, context] + shared_vars))
    else:
        result, _ = theano.scan(
            _step,
            sequences=seqs,
            outputs_info=init_states,
            non_sequences=[projected_context, context] + shared_vars,
            name=_p(prefix, '_layers'),
            n_steps=n_steps,
            profile=profile,
            strict=True,
        )

    kw_ret['hidden_without_dropout'] = result[0]
    kw_ret['memory_output'] = result[1]

    if get_gates:
        kw_ret['input_gates'] = result[4]
        kw_ret['forget_gates'] = result[5]
        kw_ret['output_gates'] = result[6]
        kw_ret['input_gates_att'] = result[7]
        kw_ret['forget_gates_att'] = result[8]
        kw_ret['output_gates_att'] = result[9]

    result = list(result)

    if dropout_params:
        result[0] = dropout_layer(result[0], *dropout_params)

    # Return memory c at the last in kw_ret
    return result[0], result[2], result[3], kw_ret


# Below are the codes for target attention
def lstm_cond_layer_v2(P, state_below, O, prefix='lstm', mask=None, context=None, one_step=False, init_memory=None,
                       init_state=None, context_mask=None, use_src_attn=False, use_trg_attn=False, **kwargs):
    layer_id = kwargs.pop('layer_id', 0)
    multi = 'multi' in O.get('unit', 'lstm_cond')
    unit_size = O.get('unit_size_cond', 2)
    # FIXME: multi-gru/lstm do NOT support get_gates now
    get_gates = kwargs.pop('get_gates', False)

    kw_ret = {}

    #assert context, 'Context must be provided'
    #assert context.ndim == 3, 'Context must be 3-d: #annotation * #sample * dim'
    if one_step:
        assert init_state, 'previous state must be provided'

    # Dimensions
    n_steps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    elif state_below.ndim == 2:
        n_samples = state_below.shape[0]
    else:
        n_samples = 1
    if multi:
        dim = P[_p(prefix, 'Wc', layer_id)][0].shape[1] // 4
    else:
        dim = P[_p(prefix, 'Wc', layer_id)].shape[1] // 4
    dropout_params = kwargs.pop('dropout_params', None)

    # Mask
    if mask is None:
        mask = T.alloc(1., n_steps, 1)

    # Initial/previous state
    if init_state is None:
        init_state = T.alloc(0., n_samples, dim)

    # Projected x
    if multi:
        state_below = T.concatenate([
            T.dot(state_below, P[_p(prefix, 'W', layer_id)][j]) + P[_p(prefix, 'b', layer_id)][j]
            for j in range(unit_size)
        ], axis=-1)
    else:
        state_below = T.dot(state_below, P[_p(prefix, 'W', layer_id)]) + P[_p(prefix, 'b', layer_id)]

    def _one_step_attention_slice(mask_, h1, c1, ctx_, Wc, U_nl, b_nl):
        preact2 = T.dot(h1, U_nl) + b_nl + T.dot(ctx_, Wc)

        i2 = T.nnet.sigmoid(_slice(preact2, 0, dim))
        f2 = T.nnet.sigmoid(_slice(preact2, 1, dim))
        o2 = T.nnet.sigmoid(_slice(preact2, 2, dim))
        c2 = T.tanh(_slice(preact2, 3, dim))

        c2 = f2 * c1 + i2 * c2
        c2 = mask_[:, None] * c2 + (1. - mask_)[:, None] * c1

        h2 = o2 * T.tanh(c2)
        h2 = mask_[:, None] * h2 + (1. - mask_)[:, None] * h1

        if get_gates:
            return h2, c2, i2, f2, o2

        return h2, c2

    def _step_slice_src(mask_, x_,
                        h_, c_, ctx_, alpha_,
                        projected_context_, context_,
                        U, Wc, W_comb_att, U_att, c_tt, U_nl, b_nl):
        # LSTM 1
        h1, c1 = _lstm_step_slice(mask_, x_, h_, c_, U)

        # Attention
        ctx_, alpha = _attention(h1, projected_context_, context_, W_comb_att, U_att, c_tt, context_mask=context_mask)

        # LSTM 2 (with attention)
        h2, c2 = _one_step_attention_slice(mask_, h1, c1, ctx_, Wc, U_nl, b_nl)

        return h2, c2, ctx_, alpha.T

    def _step_slice_trg(mask_, current_step, x_,
                        h_, c_, decoded_h, ctx_trg_,
                        trg_mask,
                        U, Wc, U_nl, b_nl,
                        Wh_trgatt, U_trgatt, b_trgatt, v_trgatt, c_trgatt):
            # LSTM 1
            h1, c1 = _lstm_step_slice(mask_, x_, h_, c_, U)


            # Attention from target
            ctx_trg_, _ = _attention_trg(h1, decoded_h,
                                         Wh_trgatt, U_trgatt, b_trgatt, v_trgatt, c_trgatt,
                                         current_step, trg_mask)

            # LSTM 2 (with attention)
            h2, c2 = _one_step_attention_slice(mask_, h1, c1, ctx_trg_, Wc, U_nl, b_nl)
            decoded_h_ret = T.set_subtensor(decoded_h[current_step+1], h2)
            return h2, c2, decoded_h_ret, ctx_trg_

    def _step_slice_srctrg(mask_, current_step, x_,
                           h_, c_, decoded_h, ctx_, ctx_trg_, alpha_,
                           projected_context_, context_, trg_mask,
                           U, Wc, W_comb_att, U_att, c_tt, U_nl, b_nl,
                           Wh_trgatt, U_trgatt, b_trgatt, v_trgatt, c_trgatt):
            # LSTM 1
            h1, c1 = _lstm_step_slice(mask_, x_, h_, c_, U)

            # Attention from source
            ctx_, alpha = _attention(h1, projected_context_, context_, W_comb_att, U_att, c_tt,
                                     context_mask=context_mask)

            # Attention from target
            ctx_trg_, _ = _attention_trg(h1, decoded_h,
                                                 Wh_trgatt, U_trgatt, b_trgatt, v_trgatt, c_trgatt,
                                                 current_step, trg_mask)

            # LSTM 2 (with attention)
            h2, c2 = _one_step_attention_slice(mask_, h1, c1, concatenate([ctx_, ctx_trg_], axis=-1), Wc, U_nl, b_nl)
            decoded_h_ret = T.set_subtensor(decoded_h[current_step+1], h2)
            return h2, c2, decoded_h_ret, ctx_, ctx_trg_, alpha.T

    def _step_slice_gates(mask_, x_,
                          h_, c_, ctx_, alpha_, i1_, f1_, o1_, i2_, f2_, o2_,
                          projected_context_, context_,
                          U, Wc, W_comb_att, U_att, c_tt, U_nl, b_nl):
        # LSTM 1
        h1, c1, i1, f1, o1 = _lstm_step_slice_gates(mask_, x_, h_, c_, i1_, f1_, o1_, U)

        # Attention
        ctx_, alpha = _attention(h1, projected_context_, context_, W_comb_att, U_att, c_tt, context_mask=context_mask)

        # LSTM 2 (with attention)
        h2, c2, i2, f2, o2 = _one_step_attention_slice(mask_, h1, c1, ctx_, Wc, U_nl, b_nl)

        return h2, c2, ctx_, alpha.T, i1, f1, o1, i2, f2, o2

    def _multi_step_slice(mask_, x_,
                          h_, c_, ctx_, alpha_,
                          projected_context_, context_,
                          U, Wc, W_comb_att, U_att, c_tt, U_nl, b_nl):
        raise Exception('Not implemented yet')

    def _build_src():
        seqs = [mask, state_below]
        init_states = [
            init_state,
            T.alloc(0., n_samples, dim) if init_memory is None else init_memory,
            T.alloc(0., n_samples, context.shape[2]),
            T.alloc(0., n_samples, context.shape[0]),
        ]
        if get_gates:
            init_states.extend([T.alloc(0., n_samples, dim) for _ in range(6)])

        shared_vars = [
            P[_p(prefix, 'U', layer_id)],
            P[_p(prefix, 'Wc', layer_id)],
            P[_p(prefix, 'W_comb_att', layer_id)],
            P[_p(prefix, 'U_att', layer_id)],
            P[_p(prefix, 'c_tt', layer_id)],
            P[_p(prefix, 'U_nl', layer_id)],
            P[_p(prefix, 'b_nl', layer_id)],
        ]
        projected_context = T.dot(context, P[_p(prefix, 'Wc_att', layer_id)]) + P[_p(prefix, 'b_att', layer_id)]
        if one_step:
            result = _step_slice_src(*(seqs + init_states + [projected_context, context] + shared_vars))
        else:
            result, _ = theano.scan(
                _step_slice_src,
                sequences=seqs,
                outputs_info=init_states,
                non_sequences=[projected_context, context] + shared_vars,
                name=_p(prefix, '_layers'),
                n_steps=n_steps,
                profile=profile,
                strict=True,
            )

        kw_ret['hidden_without_dropout'] = result[0]
        kw_ret['memory_output'] = result[1]

        if get_gates:
            kw_ret['input_gates'] = result[4]
            kw_ret['forget_gates'] = result[5]
            kw_ret['output_gates'] = result[6]
            kw_ret['input_gates_att'] = result[7]
            kw_ret['forget_gates_att'] = result[8]
            kw_ret['output_gates_att'] = result[9]

        result = list(result)

        if dropout_params:
            result[0] = dropout_layer(result[0], *dropout_params)

        # Return memory c at the last in kw_ret
        return result[0], result[2], result[3], kw_ret


    def _build_trg():
        seqs = [mask, T.arange(n_steps), state_below]
        init_states = [
            init_state,
            T.alloc(0., n_samples, dim) if init_memory is None else init_memory,
            T.alloc(0., n_steps, n_samples, dim),
            T.alloc(0., n_samples, dim),
        ]
        shared_vars = [
            P[_p(prefix, 'U', layer_id)],
            P[_p(prefix, 'Wc', layer_id)],
            P[_p(prefix, 'U_nl', layer_id)],
            P[_p(prefix, 'b_nl', layer_id)],
            P[_p(prefix, 'Wh_trgatt', layer_id)],
            P[_p(prefix, 'U_trgatt', layer_id)],
            P[_p(prefix, 'b_trgatt', layer_id)],
            P[_p(prefix, 'v_trgatt', layer_id)],
            P[_p(prefix, 'c_trgatt', layer_id)]
        ]

        if one_step:
            result = _step_slice_trg(*(seqs + init_states + [mask] + shared_vars))
        else:
            result, _ = theano.scan(
                _step_slice_trg,
                sequences=seqs,
                outputs_info=init_states,
                non_sequences=[mask] + shared_vars,
                name=_p(prefix, '_layers'),
                n_steps=n_steps,
                profile=profile,
                strict=True,
            )

        kw_ret['hidden_without_dropout'] = result[0]
        kw_ret['memory_output'] = result[1]

        if get_gates:
            kw_ret['input_gates'] = result[4]
            kw_ret['forget_gates'] = result[5]
            kw_ret['output_gates'] = result[6]
            kw_ret['input_gates_att'] = result[7]
            kw_ret['forget_gates_att'] = result[8]
            kw_ret['output_gates_att'] = result[9]

        result = list(result)

        if dropout_params:
            result[0] = dropout_layer(result[0], *dropout_params)

        # Return memory c at the last in kw_ret
        return result[0], result[3], kw_ret


    def _build_src_trg():
        seqs = [mask, T.arange(n_steps), state_below]
        init_states = [
            init_state,
            T.alloc(0., n_samples, dim) if init_memory is None else init_memory,
            T.alloc(0., n_steps, n_samples, dim),
            T.alloc(0., n_samples, context.shape[2]),
            T.alloc(0., n_samples, dim),
            T.alloc(0., n_samples, context.shape[0]),
        ]
        if get_gates:
            init_states.extend([T.alloc(0., n_samples, dim) for _ in range(6)])

        shared_vars = [
            P[_p(prefix, 'U', layer_id)],
            P[_p(prefix, 'Wc', layer_id)],
            P[_p(prefix, 'W_comb_att', layer_id)],
            P[_p(prefix, 'U_att', layer_id)],
            P[_p(prefix, 'c_tt', layer_id)],
            P[_p(prefix, 'U_nl', layer_id)],
            P[_p(prefix, 'b_nl', layer_id)],
            P[_p(prefix, 'Wh_trgatt', layer_id)],
            P[_p(prefix, 'U_trgatt', layer_id)],
            P[_p(prefix, 'b_trgatt', layer_id)],
            P[_p(prefix, 'v_trgatt', layer_id)],
            P[_p(prefix, 'c_trgatt', layer_id)]
        ]

        projected_context = T.dot(context, P[_p(prefix, 'Wc_att', layer_id)]) + P[_p(prefix, 'b_att', layer_id)]
        if one_step:
            result = _step_slice_srctrg(*(seqs + init_states + [projected_context, context, mask] + shared_vars))
        else:
            result, _ = theano.scan(
                _step_slice_srctrg,
                sequences=seqs,
                outputs_info=init_states,
                non_sequences=[projected_context, context, mask] + shared_vars,
                name=_p(prefix, '_layers'),
                n_steps=n_steps,
                profile=profile,
                strict=True,
            )

        kw_ret['hidden_without_dropout'] = result[0]
        kw_ret['memory_output'] = result[1]

        if get_gates:
            raise

        result = list(result)

        if dropout_params:
            result[0] = dropout_layer(result[0], *dropout_params)

        # Return memory c at the last in kw_ret
        return result[0], result[3], result[4], result[5], kw_ret


    if multi:
        raise Exception('Not implemented yet')
    else:
        if get_gates:
            raise Exception('Not implemented yet')

    if use_src_attn is True and use_trg_attn is False:
        return _build_src()
    elif use_src_attn is False and use_trg_attn is True:
        return _build_trg()
    elif use_src_attn is True and use_trg_attn is True:
        return _build_src_trg()
    else:
        raise Exception('Please use the lstm_layer')



def lstm_srctrgattn_layer(P, state_below, O, prefix='lstm', mask=None, context=None, one_step=False, init_memory=None,
                          init_state=None, context_mask=None, **kwargs):
    layer_id = kwargs.pop('layer_id', 0)
    multi = 'multi' in O.get('unit', 'lstm_cond')
    unit_size = O.get('unit_size_cond', 2)
    # FIXME: multi-gru/lstm do NOT support get_gates now
    get_gates = kwargs.pop('get_gates', False)

    kw_ret = {}

    assert context, 'Context must be provided'
    assert context.ndim == 3, 'Context must be 3-d: #annotation * #sample * dim'
    if one_step:
        assert init_state, 'previous state must be provided'

    # Dimensions
    n_steps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    elif state_below.ndim == 2:
        n_samples = state_below.shape[0]
    else:
        n_samples = 1
    if multi:
        dim = P[_p(prefix, 'Wc', layer_id)][0].shape[1] // 4
    else:
        dim = P[_p(prefix, 'Wc', layer_id)].shape[1] // 4
    dropout_params = kwargs.pop('dropout_params', None)

    # Mask
    if mask is None:
        mask = T.alloc(1., n_steps, 1)

    # Initial/previous state
    if init_state is None:
        init_state = T.alloc(0., n_samples, dim)

    # Projected x
    if multi:
        state_below = T.concatenate([
            T.dot(state_below, P[_p(prefix, 'W', layer_id)][j]) + P[_p(prefix, 'b', layer_id)][j]
            for j in range(unit_size)
        ], axis=-1)
    else:
        state_below = T.dot(state_below, P[_p(prefix, 'W', layer_id)]) + P[_p(prefix, 'b', layer_id)]

    def _one_step_attention_slice(mask_, h1, c1, ctx_, ctx_trg_, Wc, Wc_trg, U_nl, b_nl):
        preact2 = T.dot(h1, U_nl) + b_nl + T.dot(ctx_, Wc) + T.dot(ctx_trg_, Wc_trg)

        i2 = T.nnet.sigmoid(_slice(preact2, 0, dim))
        f2 = T.nnet.sigmoid(_slice(preact2, 1, dim))
        o2 = T.nnet.sigmoid(_slice(preact2, 2, dim))
        c2 = T.tanh(_slice(preact2, 3, dim))

        c2 = f2 * c1 + i2 * c2
        c2 = mask_[:, None] * c2 + (1. - mask_)[:, None] * c1

        h2 = o2 * T.tanh(c2)
        h2 = mask_[:, None] * h2 + (1. - mask_)[:, None] * h1

        if get_gates:
            return h2, c2, i2, f2, o2

        return h2, c2

    def _step_slice_srctrg(mask_, current_step, x_,
                           h_, c_, decoded_h, ctx_, ctx_trg_, alpha_,
                           projected_context_, context_, trg_mask,
                           U, Wc, Wc_trg, W_comb_att, U_att, c_tt, U_nl, b_nl,
                           Wh_trgatt, U_trgatt, b_trgatt, v_trgatt, c_trgatt):
            # LSTM 1
            h1, c1 = _lstm_step_slice(mask_, x_, h_, c_, U)

            # Attention from source
            ctx_, alpha = _attention(h1, projected_context_, context_, W_comb_att, U_att, c_tt,
                                     context_mask=context_mask)

            # Attention from target
            ctx_trg_, _ = _attention_trg(h1, decoded_h,
                                         Wh_trgatt, U_trgatt, b_trgatt, v_trgatt, c_trgatt,
                                         current_step, trg_mask)

            # LSTM 2 (with attention)
            h2, c2 = _one_step_attention_slice(mask_, h1, c1, ctx_, ctx_trg_, Wc, Wc_trg, U_nl, b_nl)
            decoded_h_ret = T.set_subtensor(decoded_h[current_step+1], h2)
            return h2, c2, decoded_h_ret, ctx_, ctx_trg_, alpha.T

    def _step_slice_gates(mask_, x_,
                          h_, c_, ctx_, alpha_, i1_, f1_, o1_, i2_, f2_, o2_,
                          projected_context_, context_,
                          U, Wc, W_comb_att, U_att, c_tt, U_nl, b_nl):
        # LSTM 1
        h1, c1, i1, f1, o1 = _lstm_step_slice_gates(mask_, x_, h_, c_, i1_, f1_, o1_, U)

        # Attention
        ctx_, alpha = _attention(h1, projected_context_, context_, W_comb_att, U_att, c_tt, context_mask=context_mask)

        # LSTM 2 (with attention)
        h2, c2, i2, f2, o2 = _one_step_attention_slice(mask_, h1, c1, ctx_, Wc, U_nl, b_nl)

        return h2, c2, ctx_, alpha.T, i1, f1, o1, i2, f2, o2

    def _build_src_trg():
        if one_step:
            assert init_state
            assert init_memory
            decoded_h = kwargs.pop('decoded_h', None)
            current_step = kwargs.pop('current_step', None)
            assert current_step is not None
            assert decoded_h
            assert decoded_h.ndim == 3
            seqs = [mask, current_step, state_below]
            init_states = [ init_state, init_memory, decoded_h, None, None, None]
        else:
            seqs = [mask, T.arange(n_steps), state_below]
            init_states = [
                init_state,
                T.alloc(0., n_samples, dim) if init_memory is None else init_memory,
                concatenate([init_state[None,:,:], T.zeros([n_steps, n_samples, dim], dtype=fX)],axis=0),
                ## PS: i am considering removing the first init_state !!!!!!!!!
                T.alloc(0., n_samples, context.shape[2]),
                T.alloc(0., n_samples, dim),
                T.alloc(0., n_samples, context.shape[0]),
            ]
        if get_gates:
            init_states.extend([T.alloc(0., n_samples, dim) for _ in range(6)])

        shared_vars = [
            P[_p(prefix, 'U', layer_id)],
            P[_p(prefix, 'Wc', layer_id)],
            P[_p(prefix, 'Wc_trg', layer_id)],
            P[_p(prefix, 'W_comb_att', layer_id)],
            P[_p(prefix, 'U_att', layer_id)],
            P[_p(prefix, 'c_tt', layer_id)],
            P[_p(prefix, 'U_nl', layer_id)],
            P[_p(prefix, 'b_nl', layer_id)],
            P[_p(prefix, 'Wh_trgatt', layer_id)],
            P[_p(prefix, 'U_trgatt', layer_id)],
            P[_p(prefix, 'b_trgatt', layer_id)],
            P[_p(prefix, 'v_trgatt', layer_id)],
            P[_p(prefix, 'c_trgatt', layer_id)]
        ]

        projected_context = T.dot(context, P[_p(prefix, 'Wc_att', layer_id)]) + P[_p(prefix, 'b_att', layer_id)]
        if one_step:
            result = _step_slice_srctrg(*(seqs + init_states + [projected_context, context, None] + shared_vars))
        else:
            result, _ = theano.scan(
                _step_slice_srctrg,
                sequences=seqs,
                outputs_info=init_states,
                non_sequences=[projected_context, context, mask] + shared_vars,
                name=_p(prefix, '_layers'),
                n_steps=n_steps,
                profile=profile,
                strict=True,
            )

        kw_ret['hidden_without_dropout'] = result[0]
        kw_ret['memory_output'] = result[1]

        if get_gates:
            raise

        result = list(result)

        if dropout_params:
            result[0] = dropout_layer(result[0], *dropout_params)

        # Return memory c at the last in kw_ret
        return result[0], result[3], result[4], result[5], kw_ret


    if multi:
        raise Exception('Not implemented yet')
    else:
        if get_gates:
            raise Exception('Not implemented yet')


    return _build_src_trg()


# Below are the codes for target attention
def lstm_trgattn_beforesrc(P, state_below, O, prefix='lstm', mask=None, one_step=False, init_memory=None,
                           init_state=None, **kwargs):
    layer_id = kwargs.pop('layer_id', 0)
    multi = 'multi' in O.get('unit', 'lstm_cond')
    unit_size = O.get('unit_size_cond', 2)
    # FIXME: multi-gru/lstm do NOT support get_gates now
    get_gates = kwargs.pop('get_gates', False)

    kw_ret = {}

    if one_step:
        assert init_state, 'previous state must be provided'

    # Dimensions
    n_steps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    elif state_below.ndim == 2:
        n_samples = state_below.shape[0]
    else:
        n_samples = 1
    if multi:
        dim = P[_p(prefix, 'Wc', layer_id)][0].shape[1] // 4
    else:
        dim = O['dim']
    dropout_params = kwargs.pop('dropout_params', None)

    # Mask
    if mask is None:
        mask = T.alloc(1., n_steps, 1)

    # Initial/previous state
    if init_state is None:
        init_state = T.alloc(0., n_samples, dim)

    # Projected x
    if multi:
        state_below = T.concatenate([
            T.dot(state_below, P[_p(prefix, 'W', layer_id)][j]) + P[_p(prefix, 'b', layer_id)][j]
            for j in range(unit_size)
        ], axis=-1)
    else:
        state_below = T.dot(state_below, P[_p(prefix, 'W', layer_id)]) + P[_p(prefix, 'b', layer_id)]

    def _one_step_attention_slice(mask_, h1, c1, ctx_trg_, Wc_trg, U_nl, b_nl):
        preact2 = T.dot(h1, U_nl) + b_nl + T.dot(ctx_trg_, Wc_trg)

        i2 = T.nnet.sigmoid(_slice(preact2, 0, dim))
        f2 = T.nnet.sigmoid(_slice(preact2, 1, dim))
        o2 = T.nnet.sigmoid(_slice(preact2, 2, dim))
        c2 = T.tanh(_slice(preact2, 3, dim))

        c2 = f2 * c1 + i2 * c2
        c2 = mask_[:, None] * c2 + (1. - mask_)[:, None] * c1

        h2 = o2 * T.tanh(c2)
        h2 = mask_[:, None] * h2 + (1. - mask_)[:, None] * h1

        if get_gates:
            return h2, c2, i2, f2, o2

        return h2, c2

    def _step_slice_trg(mask_, current_step, x_,
                        h_, c_, decoded_h, ctx_trg_,
                        trg_mask,
                        U, Wc, U_nl, b_nl,
                        Wh_trgatt, U_trgatt, b_trgatt, v_trgatt, c_trgatt):
            # LSTM 1
            h1, c1 = _lstm_step_slice(mask_, x_, h_, c_, U)

            # Attention from target
            ctx_trg_, _ = _attention_trg(h1, decoded_h,
                                         Wh_trgatt, U_trgatt, b_trgatt, v_trgatt, c_trgatt,
                                         current_step, trg_mask)

            # LSTM 2 (with attention)
            h2, c2 = _one_step_attention_slice(mask_, h1, c1, ctx_trg_, Wc, U_nl, b_nl)
            decoded_h_ret = T.set_subtensor(decoded_h[current_step+1], h2)
            return h2, c2, decoded_h_ret, ctx_trg_

    def _build_trg():
        if one_step:
            assert init_state
            assert init_memory
            decoded_h = kwargs.pop('decoded_h', None)
            current_step = kwargs.pop('current_step', None)
            assert current_step is not None
            assert decoded_h
            assert decoded_h.ndim == 3
            seqs = [mask, current_step, state_below]
            init_states = [init_state, init_memory, decoded_h, None]
        else:
            seqs = [mask, T.arange(n_steps), state_below]
            init_states = [
                init_state,
                T.alloc(0., n_samples, dim) if init_memory is None else init_memory,
                concatenate([init_state[None, :, :], T.zeros([n_steps, n_samples, dim], dtype=fX)], axis=0),
                T.alloc(0., n_samples, dim),
            ]
        shared_vars = [
            P[_p(prefix, 'U', layer_id)],
            P[_p(prefix, 'Wc_trg', layer_id)],
            P[_p(prefix, 'U_nl_trg', layer_id)],
            P[_p(prefix, 'b_nl_trg', layer_id)],
            P[_p(prefix, 'Wh_trgatt', layer_id)],
            P[_p(prefix, 'U_trgatt', layer_id)],
            P[_p(prefix, 'b_trgatt', layer_id)],
            P[_p(prefix, 'v_trgatt', layer_id)],
            P[_p(prefix, 'c_trgatt', layer_id)]
        ]

        if one_step:
            result = _step_slice_trg(*(seqs + init_states + [None] + shared_vars))
        else:
            result, _ = theano.scan(
                _step_slice_trg,
                sequences=seqs,
                outputs_info=init_states,
                non_sequences=[mask] + shared_vars,
                name=_p(prefix, '_layers'),
                n_steps=n_steps,
                profile=profile,
                strict=True,
            )

        kw_ret['hidden_without_dropout'] = result[0]
        kw_ret['memory_output'] = result[1]

        if get_gates:
            kw_ret['input_gates'] = result[4]
            kw_ret['forget_gates'] = result[5]
            kw_ret['output_gates'] = result[6]
            kw_ret['input_gates_att'] = result[7]
            kw_ret['forget_gates_att'] = result[8]
            kw_ret['output_gates_att'] = result[9]

        result = list(result)

        if dropout_params:
            result[0] = dropout_layer(result[0], *dropout_params)

        # Return memory c at the last in kw_ret
        return result[0], result[3], kw_ret

    if multi:
        raise Exception('Not implemented yet')
    else:
        if get_gates:
            raise Exception('Not implemented yet')

    return _build_trg()


def lstm_trgattn_aftersrc(P, state_below, O, prefix='lstm', mask=None, context_src=None, one_step=False,
                          init_memory=None, init_state=None, **kwargs):
    layer_id = kwargs.pop('layer_id', 0)
    multi = 'multi' in O.get('unit', 'lstm_cond')
    unit_size = O.get('unit_size_cond', 2)
    # FIXME: multi-gru/lstm do NOT support get_gates now
    get_gates = kwargs.pop('get_gates', False)

    kw_ret = {}

    assert context_src, 'Context must be provided'
    if one_step:
        assert context_src.ndim == 2, 'Context must be 2-d: #sample * dim'
    else:
        assert context_src.ndim == 3, 'Context must be 3-d: #annotation * #sample * dim'
    if one_step:
        assert init_state, 'previous state must be provided'

    # Dimensions
    n_steps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    elif state_below.ndim == 2:
        n_samples = state_below.shape[0]
    else:
        n_samples = 1
    if multi:
        dim = P[_p(prefix, 'Wc', layer_id)][0].shape[1] // 4
    else:
        dim = P[_p(prefix, 'Wc', layer_id)].shape[1] // 4
    dropout_params = kwargs.pop('dropout_params', None)

    # Mask
    if mask is None:
        mask = T.alloc(1., n_steps, 1)

    # Initial/previous state
    if init_state is None:
        init_state = T.alloc(0., n_samples, dim)

    # Projected x
    if multi:
        state_below = T.concatenate([
            T.dot(state_below, P[_p(prefix, 'W', layer_id)][j]) + P[_p(prefix, 'b', layer_id)][j]
            for j in range(unit_size)
        ], axis=-1)
    else:
        state_below = T.dot(state_below, P[_p(prefix, 'W', layer_id)]) + P[_p(prefix, 'b', layer_id)]

    def _one_step_attention_slice(mask_, h1, c1, ctx_, ctx_trg_, Wc, Wc_trg, U_nl_trg, b_nl_trg):
        preact2 = T.dot(h1, U_nl_trg) + b_nl_trg + T.dot(ctx_, Wc) + T.dot(ctx_trg_, Wc_trg)

        i2 = T.nnet.sigmoid(_slice(preact2, 0, dim))
        f2 = T.nnet.sigmoid(_slice(preact2, 1, dim))
        o2 = T.nnet.sigmoid(_slice(preact2, 2, dim))
        c2 = T.tanh(_slice(preact2, 3, dim))

        c2 = f2 * c1 + i2 * c2
        c2 = mask_[:, None] * c2 + (1. - mask_)[:, None] * c1

        h2 = o2 * T.tanh(c2)
        h2 = mask_[:, None] * h2 + (1. - mask_)[:, None] * h1

        if get_gates:
            return h2, c2, i2, f2, o2

        return h2, c2

    def _step_slice_trg(mask_, current_step, x_, ctx_,
                        h_, c_, decoded_h, ctx_trg_,
                        trg_mask,
                        U, Wc, Wc_trg, U_nl, b_nl,
                        Wh_trgatt, U_trgatt, b_trgatt, v_trgatt, c_trgatt):
            # LSTM 1
            h1, c1 = _lstm_step_slice(mask_, x_, h_, c_, U)

            # Attention from target
            ctx_trg_, _ = _attention_trg(h1, decoded_h,
                                         Wh_trgatt, U_trgatt, b_trgatt, v_trgatt, c_trgatt,
                                         current_step, trg_mask)

            # LSTM 2 (with attention)
            h2, c2 = _one_step_attention_slice(mask_, h1, c1, ctx_, ctx_trg_, Wc, Wc_trg, U_nl, b_nl)
            decoded_h_ret = T.set_subtensor(decoded_h[current_step+1], h2)
            return h2, c2, decoded_h_ret, ctx_trg_

    def _build_trg():
        if one_step:
            assert init_state
            assert init_memory
            decoded_h = kwargs.pop('decoded_h', None)
            current_step = kwargs.pop('current_step', None)
            assert current_step is not None
            assert decoded_h
            assert decoded_h.ndim == 3
            seqs = [mask, current_step, state_below, context_src]
            init_states = [ init_state, init_memory, decoded_h, None]
        else:
            seqs = [mask, T.arange(n_steps), state_below, context_src]
            init_states = [
                init_state,
                T.alloc(0., n_samples, dim) if init_memory is None else init_memory,
                concatenate([init_state[None,:,:], T.zeros([n_steps, n_samples, dim], dtype=fX)],axis=0),
                T.alloc(0., n_samples, dim),
            ]
        if get_gates:
            init_states.extend([T.alloc(0., n_samples, dim) for _ in range(6)])

        shared_vars = [
            P[_p(prefix, 'U', layer_id)],
            P[_p(prefix, 'Wc', layer_id)],
            P[_p(prefix, 'Wc_trg', layer_id)],
            P[_p(prefix, 'U_nl_trg', layer_id)],
            P[_p(prefix, 'b_nl_trg', layer_id)],
            P[_p(prefix, 'Wh_trgatt', layer_id)],
            P[_p(prefix, 'U_trgatt', layer_id)],
            P[_p(prefix, 'b_trgatt', layer_id)],
            P[_p(prefix, 'v_trgatt', layer_id)],
            P[_p(prefix, 'c_trgatt', layer_id)]
        ]

        if one_step:
            result = _step_slice_trg(*(seqs + init_states + [None] + shared_vars))
        else:
            result, _ = theano.scan(
                _step_slice_trg,
                sequences=seqs,
                outputs_info=init_states,
                non_sequences=[mask] + shared_vars,
                name=_p(prefix, '_layers'),
                n_steps=n_steps,
                profile=profile,
                strict=True,
            )

        kw_ret['hidden_without_dropout'] = result[0]
        kw_ret['memory_output'] = result[1]

        if get_gates:
            raise

        result = list(result)

        if dropout_params:
            result[0] = dropout_layer(result[0], *dropout_params)

        # Return memory c at the last in kw_ret
        return result[0], result[3], kw_ret


    if multi:
        raise Exception('Not implemented yet')
    else:
        if get_gates:
            raise Exception('Not implemented yet')


    return _build_trg()



__all__ = [
    'param_init_lstm',
    'lstm_layer',
    'param_init_lstm_cond',
    'lstm_cond_layer',
    'lstm_srctrgattn_layer',
    'lstm_trgattn_beforesrc',
    'lstm_trgattn_aftersrc'
]
