# -*- coding: utf-8 -*-

import numpy as np
from theano import tensor as T

from ..utility.utils import _p, normal_weight, concatenate
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


def _attention(h1, projected_context_, context_, context_mask=None, dense_attention=False, dim_word=512, dim=256, n_enc=6, *args):
    if dense_attention:
        for i in xrange(n_enc + 1):
            W_comb_att, U_att, c_tt = args[i], args[(n_enc + 1) + i], args[(n_enc + 1) * 2 + i]
            pstate_ = T.dot(h1, W_comb_att)
            if i == 0:
                pctx__ = projected_context_[:, :, : 2 * dim_word] + pstate_[None, :, :]
            else:
                pctx__ = projected_context_[:, :, 2 * (dim_word + (i - 1) * dim):2 * (dim_word + i *dim)] + pstate_[None, :, :]
            pctx__ = T.tanh(pctx__)
            alpha = T.dot(pctx__, U_att) + c_tt
            alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
            alpha = T.exp(alpha - alpha.max(axis=0, keepdims=True))
            if context_mask:
                alpha = alpha * context_mask
            alpha = alpha / alpha.sum(0, keepdims=True)
            if i == 0:
                ctx_ = (context_[:, :, :2 * dim_word] * alpha[:, :, None]).sum(0)
            else:
                _ctx_ = (context_[:, :, 2 * (dim_word + (i - 1) * dim): 2 * (dim_word + i * dim)] * alpha[:, :, None]).sum(0)
                ctx_ = concatenate([ctx_, _ctx_], axis = ctx_.ndim - 1)
    else:
        W_comb_att, U_att, c_tt = args
        pstate_ = T.dot(h1, W_comb_att)
        pctx__ = projected_context_ + pstate_[None, :, :]
        # pctx__ += xc_
        pctx__ = T.tanh(pctx__)

        alpha = T.dot(pctx__, U_att) + c_tt
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
        alpha = T.exp(alpha - alpha.max(axis=0, keepdims=True))
        if context_mask:
            alpha = alpha * context_mask
        alpha = alpha / alpha.sum(0, keepdims=True)
        ctx_ = (context_ * alpha[:, :, None]).sum(0)  # current context

    return ctx_, alpha

__all__ = [
    '_slice',
    'tanh',
    'linear',
    'dropout_layer',
    'attention_layer',
    'param_init_feed_forward',
    'feed_forward',
    '_attention',
]
