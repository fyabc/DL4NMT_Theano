#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Initialize deeper model with exist model.

Current initialize method: optimize the gap between hidden states

Origin model:
    x -> Encoder[1 layer] -> h(x) -> Decoder[1 layer] -> prob
New model:
    x -> Encoder'[2 layers] -> h'(x) -> Decoder[1 layer] -> prob'

Optimize:
    Update the parameters of Encoder' to minimize MSE loss: |h'(x) - h(x)|_2^2
    The result will be the initial value of Encoder'.
"""

from __future__ import print_function, unicode_literals

import theano.tensor as T

from layers import get_layer, embedding
from constants import profile, fX
from utils import concatenate

__author__ = 'fyabc'


def build_initializer(tparams, options):
    """Build the parameter initializer."""

    # Inputs.
    # description string: #words * #samples
    x = T.matrix('x', dtype='int64')
    x_mask = T.matrix('x_mask', dtype=fX)
    y = T.matrix('y', dtype='int64')
    y_mask = T.matrix('y_mask', dtype=fX)

    # for the backward rnn, we just need to invert x and x_mask
    xr = x[::-1]
    xr_mask = x_mask[::-1]

    n_timesteps = x.shape[0]
    n_timesteps_trg = y.shape[0]
    n_samples = x.shape[1]

    # word embedding for forward rnn (source)
    emb = embedding(tparams, x, options, n_timesteps, n_samples)
    proj = get_layer(options['encoder'])[1](tparams, emb, options,
                                            prefix='encoder',
                                            mask=x_mask)
    # word embedding for backward rnn (source)
    embr = embedding(tparams, xr, options, n_timesteps, n_samples)
    projr = get_layer(options['encoder'])[1](tparams, embr, options,
                                             prefix='encoder_r',
                                             mask=xr_mask)

    # context will be the concatenation of forward and backward rnns
    ctx = concatenate([proj[0], projr[0][::-1]], axis=proj[0].ndim - 1)


def main():
    pass


if __name__ == '__main__':
    main()
