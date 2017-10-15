#! /usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np

try:
    from bottleneck import argpartsort as arg_top_k
except ImportError:
    from bottleneck import argpartition
    from functools import partial

    def arg_top_k(arr, int_n ,axis=-1):
        return argpartition(arr, int_n - 1, axis=axis)

from ..constants import fX

__author__ = 'fyabc'


def floatX(value):
    return np.asarray(value, dtype=fX)


__all__ = [
    'floatX',
    'arg_top_k',
]
