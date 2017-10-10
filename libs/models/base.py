#! /usr/bin/python
# -*- coding: utf-8 -*-

from collections import OrderedDict

import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from libs.utility.utils import _p
from ..layers import tanh
from ..utility.utils import _p
from ..constants import fX


class NMTModelBase(object):
    """Base class of NMT models.

    Contains some common used layers and methods.
    """

    def __init__(self, options, given_params=None):
        # Dict of options
        self.O = options

        # Dict of parameters (Theano shared variables)
        self.P = OrderedDict() if given_params is None else given_params

        # Dict of duplicated parameters (for multiverso)
        self.dupP = OrderedDict()

    @staticmethod
    def get_input():
        """Get model input.

        Model input shape: #words * #samples

        Returns
        -------
        x, x_mask, y, y_mask: Theano tensor variables
        """

        x = T.matrix('x', dtype='int64')
        x_mask = T.matrix('x_mask', dtype=fX)
        y = T.matrix('y', dtype='int64')
        y_mask = T.matrix('y_mask', dtype=fX)

        return x, x_mask, y, y_mask

    @staticmethod
    def input_dimensions(x, y=None):
        """Get input dimensions.

        Parameters
        ----------
        x: Theano tensor
            Source sentence batch
        y: Theano tensor or None
            Target sentence batch

        Returns
        -------
        3 Theano variables:
            n_timestep, n_timestep_tgt, n_samples
        """

        n_timestep = x.shape[0]
        n_timestep_tgt = y.shape[0] if y else None
        n_samples = x.shape[1]

        return n_timestep, n_timestep_tgt, n_samples

    @staticmethod
    def reverse_input(x, x_mask):
        return x[::-1], x_mask[::-1]

    def _dropout_params(self, trng=None, use_noise=None):
        dropout_rate = self.O['use_dropout']
        trng = RandomStreams(1234) if trng is None else trng
        use_noise = theano.shared(np.float32(0.)) if use_noise is None else use_noise

        if dropout_rate is not False:
            return trng, use_noise, [use_noise, trng, dropout_rate]
        else:
            return trng, use_noise, None

    def embedding(self, input_, n_timestep, n_samples, emb_name='Wemb'):
        """Embedding layer: input -> embedding"""

        emb = self.P[emb_name][input_.flatten()]
        emb = emb.reshape([n_timestep, n_samples, self.O['dim_word']])

        return emb

    @staticmethod
    def dropout(input_, use_noise, trng, dropout_rate = 0.):
        """Dropout"""

        projection = T.switch(
            use_noise,
            input_ * trng.binomial(input_.shape, p=(1. - dropout_rate), n=1,
                                   dtype=input_.dtype),
            input_ * (1. - dropout_rate))
        return projection

    def feed_forward(self, input_, prefix, activation=tanh):
        """Feed-forward layer."""

        if isinstance(activation, (str, unicode)):
            activation = eval(activation)
        return activation(T.dot(input_, self.P[_p(prefix, 'W')]) + self.P[_p(prefix, 'b')])

    @staticmethod
    def get_context_mean(context, x_mask):
        """Get mean of context (across time) as initial state of decoder RNN

        Or you can use the last state of forward + backward encoder RNNs
            # return concatenate([proj[0][-1], projr[0][-1]], axis=proj[0].ndim-2)
        """

        return (context * x_mask[:, :, None]).sum(0) / x_mask.sum(0)[:, None]

    def attention_projected_context(self, context, prefix='lstm', **kwargs):
        attention_layer_id = self.O['attention_layer_id']
        pre_projected_context_ = T.dot(context, self.P[_p(prefix, 'Wc_att', attention_layer_id)]) + \
            self.P[_p(prefix, 'b_att', attention_layer_id)]
        return pre_projected_context_
