#! /usr/bin/python
# -*- coding: utf-8 -*-

"""The Theano operator of arg-top-k."""

import theano
import theano.tensor as T

from .basic import arg_top_k

__author__ = 'fyabc'


class ArgPartSortOp(theano.Op):
    """
    This class is a wrapper of numpy bottleneck arg-top-k function.

    """

    __props__ = ('k',)

    def __init__(self, k):
        self.k = k

    def __str__(self):
        return self.__class__.__name__ + '{%d}' % self.k

    def make_node(self, input_, axis=-1):
        input_ = T.as_tensor_variable(input_)
        axis = T.as_tensor_variable(axis)
        bcast = input_.type.broadcastable
        return theano.Apply(self, [input_, axis], [T.TensorType(dtype='int64', broadcastable=bcast)()])

    def perform(self, node, inputs, output_storage, params=None):
        a = inputs[0]
        axis = inputs[1]
        z = output_storage[0]
        z[0] = theano._asarray(arg_top_k(a, self.k, axis), dtype=node.outputs[0].dtype)

    def infer_shape(self, node, inputs_shapes):
        if (isinstance(node.inputs[1], theano.Constant) and
                node.inputs[1].data is None):
            return [(T.mul(*inputs_shapes[0]),)]
        # axis should not be None, so there should be the same number of
        # dimensions in the input and output
        assert node.inputs[0].ndim == node.outputs[0].ndim
        assert inputs_shapes[1] == ()
        return [inputs_shapes[0]]

    def grad(self, inputs, outputs_grads):
        # No grad defined for integers.
        inp, axis = inputs
        inp_grad = inp.zeros_like()
        axis_grad = theano.gradient.grad_undefined(
            self, 1, axis,
            "argpartsort is not defined for non-integer axes so"
            " argpartsort(x, axis+eps) is undefined")
        return [inp_grad, axis_grad]

    def R_op(self, inputs, eval_points):
        raise NotImplementedError('R-op not implemented for argpartsort')


def theano_argpartsort(a, k, axis=-1):
    """
    The Theano operator of arg-top-k.
    Parameters
    ----------
    a
    k
    axis

    Returns
    -------

    """
    if axis is None:
        a = a.flatten()
        axis = 0
    return ArgPartSortOp(k)(a, axis)


__all__ = [
    'arg_top_k',
    'theano_argpartsort',
]
