#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Training utilities."""

import numpy as np

from .utils import prepare_data, get_batch_place_holder


def get_train_input(*args, **kwargs):
    if kwargs.pop('placeholder', False):
        x, x_mask, y, y_mask = get_batch_place_holder(kwargs['batch_size'], kwargs['maxlen'])
    else:
        x, x_mask, y, y_mask = prepare_data(*args, maxlen=kwargs['maxlen'])

    if x is None:
        return None

    use_delib = kwargs['use_delib']
    if use_delib:
        y_pos_ = np.repeat(np.arange(y.shape[0])[:, None], y.shape[1], axis=1).astype('int64')
        which_word = kwargs.pop('which_word', None)
        if which_word is not None:
            try:
                y = y[which_word, None]
                y_mask = y_mask[which_word, None]
                y_pos_ = y_pos_[which_word, None]
            except:
                return None
    inputs = [x, x_mask, y, y_mask]
    if use_delib:
        inputs.append(y_pos_)

    return inputs


__all__ = [
    'get_train_input',
]
