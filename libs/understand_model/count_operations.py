#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Count operations (dot, elemwise multiply, etc.) of the model."""

from __future__ import print_function

import numpy as np
import theano
import theano.tensor as T

from ..utility.utils import load_options_test, message, print_params
from ..constants import fX
from ..models import build_and_init_model


class OpCounter(object):
    def __init__(self, args):
        # About model
        model_name = args.modelpath
        self.O = load_options_test(model_name)

        model_type = 'NMTModel'
        if self.O['trg_attention_layer_id'] is not None:
            model_type = 'TrgAttnNMTModel'
        self.model, _, ret = build_and_init_model(model_name, self.O, build=True, model_type=model_type)

        print_params(self.model.P)

        trng, use_noise, \
            x, x_mask, y, y_mask, \
            opt_ret, \
            cost, test_cost, x_emb = ret
        inps = [x, x_mask, y, y_mask]

        if args.dest == 'probs':
            self.f = theano.function(
                inps, cost,
                profile=False,
                mode=theano.compile.MonitorMode(
                    pre_func=self.inspect_inputs,
                    post_func=self.inspect_outputs,
                )
            )

        self.fake_inputs = self._get_fake_inputs()

        # Counters
        self.dots = []
        self.elemwises = []
        self.n_nodes = 0

    def inspect_inputs(self, i, node, fn):
        message('Index:', i, 'Node:', node)
        message('inputs:')
        for input_ in fn.inputs:
            message('\t shape: {} dtype: {}'.format(input_[0].shape, input_[0].dtype))

        op_name = type(node.op).__name__.lower()
        if 'dot' in op_name:
            self.dots.append([i] + [input_[0].shape for input_ in fn.inputs])
        if 'elemwise' in op_name:
            self.elemwises.append([i, node.op.scalar_op] + [input_[0].shape for input_ in fn.inputs])
        if i > self.n_nodes:
            self.n_nodes = i

    def inspect_outputs(self, i, node, fn):
        message('outputs:')
        for output in fn.outputs:
            message('\t shape: {} dtype: {}'.format(output[0].shape, output[0].dtype))
        message()

    def run(self):
        self.f(**self.fake_inputs)

    def _get_fake_inputs(self):
        maxlen = self.O['maxlen']
        maxlen_x = min(maxlen, 10)
        maxlen_y = min(maxlen, 8)
        n_samples = self.O['batch_size']
        n_words_src = self.O['n_words_src']
        n_words = self.O['n_words']

        return {
            'x': np.random.randint(0, n_words_src, (maxlen_x, n_samples), dtype='int64'),
            'y': np.random.randint(0, n_words, (maxlen_y, n_samples), dtype='int64'),
            'x_mask': np.ones((maxlen_x, n_samples), dtype=fX),
            'y_mask': np.ones((maxlen_y, n_samples), dtype=fX),
        }

    def report(self):
        message('Number of nodes:', self.n_nodes)
        message('Dots:')
        for record in self.dots:
            message('\tindex: {} shapes: {}'.format(record[0], ' '.join(str(s) for s in record[1:])))
        message('Elemwises:')
        for record in self.elemwises:
            message('\tindex: {} scalar_op: {} shapes: {}'.format(
                record[0], record[1], ' '.join(str(s) for s in record[2:])))


def real_main(args):
    counter = OpCounter(args)

    message('Inputs:')
    for k, v in counter.fake_inputs.items():
        message('\t {} shape: {} dtype: {}'.format(k, v.shape, v.dtype))
    message()

    counter.run()

    counter.report()
