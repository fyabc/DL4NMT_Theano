#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Initialize deeper model with exist model using regression.

Current initialize method: optimize the gap between hidden states

Origin model:
    x -> Encoder[1 layer] -> h(x) -> Decoder[1 layer] -> prob
New model:
    x -> Encoder'[2 layers] -> h'(x) -> Decoder[1 layer] -> prob'

Optimize:
    Update the parameters of Encoder' to minimize MSE loss: |h'(x) - h(x)|_2^2
    The result will be the initial value of Encoder'.
"""

from __future__ import print_function

import argparse
from collections import OrderedDict

import theano.tensor as T
import numpy as np

from model import NMTModel
from config import DefaultOptions
from data_iterator import TextIterator
from utils import load_options, itemlist, apply_gradient_clipping, make_f_train

__author__ = 'fyabc'


def combine_parameters(old_parameters, new_parameters):
    combined = old_parameters.copy()

    for k, v in new_parameters.iteritems():
        combined['new_{}'.format(k)] = v

    return combined


def build_regression(args, top_options):
    """The main function to build the regression.

    :param args: Options from the argument parser
    :param top_options: Options from top-level (like options in train_nmt.py)
    """

    # Initialize and load options.
    old_options = DefaultOptions.copy()
    old_options.update(top_options)
    load_options(old_options)

    # Initialize options of new model.
    new_options = old_options.copy()
    new_options['n_encoder_layers'] = args.n_encoder_layers
    new_options['n_decoder_layers'] = args.n_decoder_layers

    only_encoder = new_options['n_decoder_layers'] == old_options['n_decoder_layers']

    # Old and new models.
    old_model = NMTModel(old_options)
    new_model = NMTModel(new_options)

    if only_encoder:
        # Initialize and reload model parameters.
        print('Initializing and reloading model parameters...', end='')
        old_model.initializer.init_input_to_context(old_model.P)
        new_model.initializer.init_input_to_context(new_model.P)
        print('Done')

        # Build model.
        print('Building model...', end='')
        input_, context_old = old_model.input_to_context()
        _, context_new = new_model.input_to_context(input_)
        print('Done')

        # Build MSE loss.
        loss = ((context_new - context_old) ** 2).mean()

        # Combine parameters
        combined_parameters = combine_parameters(old_model.P, new_model.P)

        # Compute gradient.
        print('Computing gradient...', end='')
        grads = T.grad(loss, wrt=itemlist(combined_parameters))
        grads = apply_gradient_clipping(old_options, grads)
        print('Done')

        # Build optimizer.
        print('Building optimizers...', end='')
        lr = T.scalar(name='lr')
        f_grad_shared, f_update = eval(args.regression_optimizer)(lr, combined_parameters, grads, input_, loss)
        f_train = make_f_train(f_grad_shared, f_update)
        print('Done')

        print('Loading data...', end='')
        text_iterator = TextIterator(
            old_options['datasets'][0],
            old_options['datasets'][1],
            old_options['vocab_filenames'][0],
            old_options['vocab_filenames'][1],
            old_options['batch_size'],
            old_options['maxlen'],
            old_options['n_words_src'],
            old_options['n_words'],
        )
        print('Done')

        print('Optimization')
        # todo
    else:
        raise Exception('Do not support decoder regression now, need to be implemented')


def main():
    parser = argparse.ArgumentParser(description='The regression task to initialize the model.')
    parser.add_argument('--enc', action='store', default=2, type=int, dest='n_encoder_layers',
                        help='Number of encoder layers of new model, default is 2')
    parser.add_argument('--dec', action='store', default=1, type=int, dest='n_decoder_layers',
                        help='Number of decoder layers of new model, default is 1')
    parser.add_argument('--lr', action="store", metavar="learning_rate", dest="learning_rate", type=float, default=0.8)
    parser.add_argument('model_file', nargs='?', default='model/baseline/baseline.npz',
                        help='Generated model file, default is "model/baseline/baseline.npz"')
    parser.add_argument('pre_load_file', nargs='?', default='model/en2fr.iter160000.npz',
                        help='Pre-load model file, default is "model/en2fr.iter160000.npz"')
    # todo: Add more arguments

    args = parser.parse_args()

    build_regression(args, dict(
        saveto=args.model_file,
        preload=args.pre_load_file,
        reload_=args.reload,
        dim_word=620,
        dim=1000,
        decay_c=0.,
        clip_c=1.,
        lrate=args.learning_rate,
        optimizer=args.optimizer,
        patience=1000,
        maxlen=1000,
        batch_size=80,
        dispFreq=2500,
        saveFreq=10000,
        datasets=[r'.\data\train\filtered_en-fr.en',
                  r'.\data\train\filtered_en-fr.fr'],
        use_dropout=False,
        overwrite=False,
        # picked_train_idxes_file=params['train_idx_file'],
        n_words=30000,
        n_words_src=30000,
        sort_by_len=args.curri,

        # Options from v-yanfa
        convert_embedding=args.convert_embedding,
        dump_before_train=args.dump_before_train,
        plot_graph=args.plot,

        # [NOTE]: This is for old model, settings for new model are in args
        n_encoder_layers=1,
        n_decoder_layers=1,
    ))


if __name__ == '__main__':
    main()
