#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Convert the exist models to new models."""

from __future__ import print_function

import sys
import numpy as np

__author__ = 'fyabc'


def encoder_to_whole(encoder_model, whole=None, save_filename=None):
    """Fill the encoder weights to the whole model.

    Just use once.

    :param encoder_model:
    :param whole:
    :param save_filename:
    :return:
    """

    whole = 'model/en2fr_iteration160000.npz' if whole is None else whole
    save_filename = encoder_model if save_filename is None else save_filename

    old_params = np.load(whole)

    encoder_params = np.load(encoder_model)

    old_params.update(encoder_params)

    np.savez(save_filename, **old_params)


def main():
    encoder_model = sys.argv[1]

    for i in sys.argv[2:]:
        print('Converting iteration {}...'.format(i), end='')
        encoder_to_whole('init_')
        print('Done')


if __name__ == '__main__':
    main()
