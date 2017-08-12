# -*- coding: utf-8 -*-

import numpy as np
from theano import tensor as T

from ..utility.utils import _p, normal_weight
from ..constants import fX

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

def layer_normalization_layer(z, alpha, beta, eps=1e-6):
    '''
    refer to https://arxiv.org/pdf/1607.06450.pdf, Eqn.(15) ~ Eqn.(22)
    :param z: (..., dim)
    :param alpha: (dim,)
    :param beta: (dim,)
    :return: normalized matrix
    '''
    mu_ = z.mean(axis=-1, keepdims=True)
    var_ = T.var(z, axis=-1, keepdims=True)
    return ((z - mu_) / T.sqrt(var_ + eps)) * alpha + beta


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


def _attention(h1, projected_context_, context_, W_comb_att, U_att, c_tt, context_mask=None):
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

    return ctx_, alpha


def _attention_trg(h1, decoded_h,
                   Wh_trgatt, U_trgatt, b_trgatt, v_trgatt, c_trgatt,
                   current_step, trg_mask=None):
    decoded_h_ = decoded_h[:(current_step + 1)]
    pstate_trg = T.dot(h1, Wh_trgatt) + T.dot(decoded_h_, U_trgatt) + b_trgatt

    pstate_trg = T.dot(T.tanh(pstate_trg), v_trgatt) + c_trgatt
    pstate_trg = pstate_trg.reshape([pstate_trg.shape[0], pstate_trg.shape[1]])
    exp_pstate_trg = T.exp(pstate_trg)
    if trg_mask:
        exp_pstate_trg *= trg_mask[:(current_step+1)]
    alpha_trg = exp_pstate_trg / exp_pstate_trg.sum(axis=0, keepdims=True)
    ctx_trg_ = (decoded_h_ * alpha_trg[:, :, None]).sum(axis=0)
    return ctx_trg_, alpha_trg


__all__ = [
    '_slice',
    'tanh',
    'linear',
    'dropout_layer',
    'layer_normalization_layer',
    'attention_layer',
    'param_init_feed_forward',
    'feed_forward',
    '_attention',
    '_attention_trg',
]
