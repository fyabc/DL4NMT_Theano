#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import numpy as np

import theano
import theano.tensor as T

from ..utility.utils import normal_weight
from ..constants import fX

def inspect_inputs(i, node, fn):
    print('Index:', i, 'Node:', node)
    print('inputs:')
    for input_ in fn.inputs:
        print('\t shape: {} dtype: {}'.format(input_[0].shape, input_[0].dtype))


def inspect_outputs(i, node, fn):
    print('outputs:')
    for output in fn.outputs:
        print('\t shape: {} dtype: {}'.format(output[0].shape, output[0].dtype))
    print()


def test_func(grad=True):
    x = T.matrix('x', dtype=fX)
    y = T.matrix('y', dtype=fX)
    a = theano.shared(normal_weight(3, 5))

    z = T.dot(x, y)
    o = (z + 1) * a

    if grad:
        output = T.grad(o.mean(), [x, y])
    else:
        output = o

    return [x, y], [a], output


def real_main(args):
    inputs, shares, output = test_func(True)

    f = theano.function(
        inputs, output,
        mode=theano.compile.MonitorMode(
            pre_func=inspect_inputs,
            post_func=inspect_outputs,
        )
    )

    f(normal_weight(3, 4), normal_weight(4, 5))
