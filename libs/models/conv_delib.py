#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Convolutional deliberation network"""

from __future__ import print_function

import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from ..layers.basic import _slice
from .model import ParameterInitializer, NMTModel
from ..utility.utils import *
from ..layers import *
from ..constants import fX, profile, AverageLength


class ConvDelibInitializer(ParameterInitializer):
    pass


class ConvDelib(NMTModel):
    def __init__(self, options, given_params=None):
        super(ConvDelib, self).__init__(options, given_params)
        self.initializer = ConvDelibInitializer(options)

    def conv_decoder(self, tgt_pos_embed, context, x_mask, y=None, y_mask=None, **kwargs):
        """Convolutional decoder.

        Parameters
        ----------
        tgt_pos_embed: Tensor, ([Tt], [Bs], [W])
        context: Tensor, ([Ts], [Bs], [Hc])
        x_mask: Tensor, ([Ts], [Bs])
        y: Tensor, ([Tt], [Bs])
        y_mask: Tensor, ([Tt], [Bs])
        kwargs

        Returns
        -------
        probs: Tensor, ([Tt] * [Bs], [n_words])
            Probabilities of target sentence
        """

        pass

    def build_model(self, set_instance_variables=False):
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

        probs = self.conv_decoder(tgt_pos_embed, context, x_mask, y, y_mask, trng=trng, use_noise=use_noise)

        test_cost = self.build_cost(y, y_mask, probs)
        cost = test_cost / self.O['cost_normalization']  # cost used to derive gradient in training

        return trng, use_noise, x, x_mask, y, y_mask, y_pos_, opt_ret, cost, test_cost, probs
