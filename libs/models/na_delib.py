#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Deliberation model with Non-AutoRegressive first-pass decoder."""

import theano.tensor as T

from .model import NMTModel

__author__ = 'fyabc'


class NADelibModel(NMTModel):
    def __init__(self, options, given_params=None):
        super(NADelibModel, self).__init__(options, given_params)

    def build_model(self, set_instance_variables=False):
        """Build a training model."""

        opt_ret, trng, use_noise, dropout_params = self._prepare_encoder()

        (x, x_mask, y, y_mask), context, _ = self.input_to_context(dropout_params=dropout_params)

        n_timestep, n_timestep_tgt, n_samples = self.input_dimensions(x, y)

        context_mean = self.get_context_mean(context, x_mask)

        tgt_embedding = self.embedding(y, n_timestep_tgt, n_samples, 'Wemb_dec')
        tgt_embedding += self._pos_embedding(n_timestep_tgt, n_samples, 'Wemb_dec_pos')
        tgt_embedding_shifted = self._shift_embedding(tgt_embedding)

        pre_projected_context = self.attention_projected_context(context, prefix='decoder')

        na_decoder_output = self._first_pass_decoder(None)

        # todo: RNN second-pass decoder

    def _first_pass_decoder(self, encoder_output):
        # todo: NA first-pass decoder
        return encoder_output
