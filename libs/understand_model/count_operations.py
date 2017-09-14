#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Count operations (dot, elemwise multiply, etc.) of the model."""

from __future__ import print_function

import numpy as np
import theano
import theano.tensor as T

from ..utility.utils import normal_weight
from ..constants import fX


class OpCounter(object):
    def __init__(self, O):
        # About model

        # Counters
        self.dots = []

    def inspect_inputs(self, i, node, fn):
        print('Index:', i, 'Node:', node)
        print('inputs:')
        for input_ in fn.inputs:
            print('\t shape: {} dtype: {}'.format(input_[0].shape, input_[0].dtype))

        op_name = type(node.op).__name__.lower()
        if 'dot' in op_name:
            self.dots.append([input_[0].shape for input_ in fn.inputs])

    def inspect_outputs(self, i, node, fn):
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

    w = theano.shared(np.random.randn(2, 5, 7).astype(fX))

    o = T.dot(o, w)

    if grad:
        output = T.grad(o.mean(), [x, y])
    else:
        output = o

    return [x, y], [a], output


def real_main(args=None):
    inputs, shares, output = test_func(True)
    counter = OpCounter({})

    f = theano.function(
        inputs, output,
        mode=theano.compile.MonitorMode(
            pre_func=counter.inspect_inputs,
            post_func=counter.inspect_outputs,
        )
    )

    f(normal_weight(3, 4), normal_weight(4, 5))

    print(counter.dots)

    # todo
