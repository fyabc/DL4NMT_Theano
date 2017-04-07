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

import sys
import argparse
import time

import theano
import theano.tensor as T
import numpy as np

from model import NMTModel, ParameterInitializer
from config import DefaultOptions
from data_iterator import TextIterator
from utils import *
from optimizers import Optimizers

__author__ = 'fyabc'


def build_loss(x, x_mask, context_old, context_new, args):
    # Context shape: ([Ts], [BS], [Hc]))
    # Mask shape: ([Ts], [BS])

    f_context_old = theano.function([x, x_mask], context_old)
    f_context_new = theano.function([x, x_mask], context_new)

    delta_context = (context_old - context_new) * x_mask[:, :, None]
    loss = (delta_context ** 2).sum() / delta_context.shape[0]
    f_loss = theano.function([x, x_mask], loss, profile=False)

    return f_context_old, f_context_new, loss, f_loss


def build_decoder_loss(
        x, x_mask, y, y_mask,
        hidden_decoder_old, hidden_decoder_new, context_decoder_old, context_decoder_new,
        args):
    # Hidden shape: ([Tt], [BS], [H])
    # Context shape: ([Ts], [BS], [Hc])
    # Mask shape: ([Ts/t], [BS])

    f_context_old = theano.function([x, x_mask, y, y_mask], context_decoder_old)
    f_context_new = theano.function([x, x_mask, y, y_mask], context_decoder_new)

    delta_context = (context_decoder_old - context_decoder_new) * y_mask[:, :, None]
    delta_hidden = (hidden_decoder_old - hidden_decoder_new) * y_mask[:, :, None]

    loss = (delta_context ** 2).sum() / delta_context.shape[0] + (delta_hidden ** 2).sum() / delta_hidden.shape[0]

    f_loss = theano.function([x, x_mask, y, y_mask], loss, profile=False)

    return f_context_old, f_context_new, loss, f_loss


def validate(iterator, small_train_iterator, f_loss, only_encoder=True, maxlen=None):
    valid_cost = 0.0
    count = 0

    for x, y in iterator:
        x, x_mask, y, y_mask = prepare_data(x, y, maxlen=maxlen)

        if x is None:
            continue

        inputs = [x, x_mask] if only_encoder else [x, x_mask, y, y_mask]

        valid_cost += f_loss(*inputs)
        count += 1

    valid_cost /= count

    print('Valid average loss per batch: {}'.format(valid_cost))

    cost = 0.0
    count = 0

    for x, y in small_train_iterator:
        x, x_mask, y, y_mask = prepare_data(x, y, maxlen=maxlen)

        if x is None:
            continue

        inputs = [x, x_mask] if only_encoder else [x, x_mask, y, y_mask]

        cost += f_loss(*inputs)
        count += 1

    cost /= count

    print('Small train average loss per batch: {}'.format(cost))

    return valid_cost


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
    new_options['encoder_many_bidirectional'] = args.connection_type == 1

    only_encoder = new_options['n_decoder_layers'] == old_options['n_decoder_layers']

    # Old and new models.
    old_model = NMTModel(old_options)
    new_model = NMTModel(new_options)

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

    small_train_iterator = TextIterator(
        old_options['small_train_datasets'][0],
        old_options['small_train_datasets'][1],
        old_options['vocab_filenames'][0],
        old_options['vocab_filenames'][1],
        old_options['batch_size'],
        old_options['maxlen'],
        old_options['n_words_src'],
        old_options['n_words'],
    )

    valid_text_iterator = TextIterator(
        old_options['valid_datasets'][0],
        old_options['valid_datasets'][1],
        old_options['vocab_filenames'][0],
        old_options['vocab_filenames'][1],
        old_options['valid_batch_size'],
        old_options['maxlen'],
        old_options['n_words_src'],
        old_options['n_words'],
    )
    print('Done')

    if only_encoder:
        f_initialize = ParameterInitializer.init_input_to_context
    else:
        f_initialize = ParameterInitializer.init_input_to_decoder_context

    # Initialize and reload model parameters.
    print('Initializing and reloading model parameters...', end='')
    f_initialize(old_model.initializer, old_model.P, reload_=True)
    f_initialize(
        new_model.initializer, new_model.P,
        reload_=args.warm_start_file is not None,
        preload=args.warm_start_file,
    )
    print('Done')

    # Build model.
    if only_encoder:
        print('Building model...', end='')
        input_, context_old = old_model.input_to_context()
        x, x_mask, y, y_mask = input_
        _, context_new = new_model.input_to_context(input_)
        print('Done')

        # Build output and MSE loss.
        f_context_old, f_context_new, loss, f_loss = build_loss(x, x_mask, context_old, context_new, args)
    else:
        print('Building model...', end='')
        input_, hidden_decoder_old, context_decoder_old = old_model.input_to_decoder_context()
        x, x_mask, y, y_mask = input_
        _, hidden_decoder_new, context_decoder_new = new_model.input_to_decoder_context(input_)
        print('Done')

        f_context_old, f_context_new, loss, f_loss = build_decoder_loss(
            x, x_mask, y, y_mask,
            hidden_decoder_old, hidden_decoder_new, context_decoder_old, context_decoder_new, args)

    # Compute gradient.
    print('Computing gradient...', end='')
    trainable_parameters = new_model.P.copy()
    if args.fix_embedding:
        print('Fix word embedding!')
        del trainable_parameters['Wemb']

        if not only_encoder:
            del trainable_parameters['Wemb_dec']

    # Build L2 regularization.
    l2_regularization(loss, trainable_parameters, args.decay_c)

    grads = T.grad(loss, wrt=itemlist(trainable_parameters))

    # Apply gradient clipping.
    apply_gradient_clipping(args.clip_c, grads)
    print('Done')

    # Build optimizer.
    inputs = [x, x_mask] if only_encoder else [x, x_mask, y, y_mask]

    # Build optimizer.
    print('Building optimizers...', end='')
    lr = T.scalar(name='lr')
    f_grad_shared, f_update = Optimizers[args.regression_optimizer](
        lr, trainable_parameters, grads, inputs, loss)
    print('Done')

    print('Optimization')
    start_time = time.time()
    iteration = 0
    estop = False

    if args.dump_before_train:
        print('Dumping before train...', end='')
        new_model.save_whole_model(args.model_file, iteration)
        print('Done')

    # Validate before train
    validate(valid_text_iterator, small_train_iterator, f_loss, only_encoder, top_options['maxlen'])

    learning_rate = args.learning_rate

    for epoch in xrange(args.max_epoch):
        n_samples = 0

        for i, (x, y) in enumerate(text_iterator):
            n_samples += len(x)
            iteration += 1

            x, x_mask, y, y_mask = prepare_data(x, y, maxlen=top_options['maxlen'])

            if x is None:
                print('Minibatch with zero sample under length ', top_options['maxlen'])
                iteration -= 1
                continue

            inputs = [x, x_mask] if only_encoder else [x, x_mask, y, y_mask]

            if args.debug:
                print('Cost before train: {}'.format(f_loss(*inputs)))

            # Train!
            cost = f_grad_shared(*inputs)
            f_update(learning_rate)

            if args.debug:
                print('Cost after train: {}'.format(f_loss(*inputs)))

            if np.isnan(cost) or np.isinf(cost):
                print('NaN detected')
                return 1., 1., 1.

            # verbose
            if np.mod(iteration, args.disp_freq) == 0:
                print('Epoch {} Update {} Cost {:.6f} Time {:.6f}min'.format(
                    epoch, iteration, float(cost), (time.time() - start_time) / 60.0,
                ))
                sys.stdout.flush()

            if np.mod(iteration, args.save_freq) == 0:
                new_model.save_whole_model(args.model_file, iteration)

            if np.mod(iteration, args.valid_freq) == 0:
                validate(valid_text_iterator, small_train_iterator, f_loss, only_encoder, top_options['maxlen'])

                if args.debug and args.dump_hidden is not None:
                    print('Dumping input and hidden state to {}...'.format(args.dump_hidden), end='')
                    np.savez(
                        args.dump_hidden,
                        x=x, x_mask=x_mask,
                        y=y, y_mask=y_mask,
                        hidden_old=f_context_old(*inputs),
                        hidden_new=f_context_new(*inputs),
                    )
                    print('Done')

            if args.discount_lr_freq > 0 and np.mod(iteration, args.discount_lr_freq) == 0:
                learning_rate *= 0.5
                print('Discount learning rate to {}'.format(learning_rate))

            # finish after this many updates
            if iteration >= args.finish_after:
                print ('Finishing after {} iterations!'.format(iteration))
                estop = True
                break

        print('Seen {} samples'.format(n_samples))

        if estop:
            break

    return 0.


def main():
    parser = argparse.ArgumentParser(description='The regression task to initialize the model.')
    # parser.add_argument('-r', action="store_true", default=False, dest='reload',
    #                     help='Reload, default to False, set to True')
    parser.add_argument('--enc', action='store', default=2, type=int, dest='n_encoder_layers',
                        help='Number of encoder layers of new model, default is 2')
    parser.add_argument('--dec', action='store', default=1, type=int, dest='n_decoder_layers',
                        help='Number of decoder layers of new model, default is 1')
    parser.add_argument('--lr', action="store", metavar="learning_rate", dest="learning_rate", type=float, default=0.1)
    parser.add_argument('model_file', nargs='?', default='model/init/en2fr_init_encoder2.npz',
                        help='Generated model file, default is "model/init/en2fr_init_encoder2.npz"')
    parser.add_argument('pre_load_file', nargs='?', default='model/en2fr.iter160000.npz',
                        help='Pre-load model file (to old model, the optimize target), '
                             'default is "model/en2fr.iter160000.npz"')
    parser.add_argument('-w', '--warm_start_file', action='store', default=None, dest='warm_start_file',
                        help='The warm start model file (to new model), default is None (cold start)')
    parser.add_argument('-D', action='store_false', default=True, dest='dump_before_train',
                        help='Dump before train default to True, set to False')
    parser.add_argument('--optimizerR', action='store', default='adam', dest='regression_optimizer',
                        help='Regression optimizer, default is "adam"')
    parser.add_argument('-e', '--epoch', action='store', default=5000, type=int, dest='max_epoch',
                        help='The max epoch of regression, default is 5000')
    parser.add_argument('--disp_freq', action='store', default=50, type=int, dest='disp_freq',
                        help='The display frequency, default is 10')
    parser.add_argument('--save_freq', action='store', default=10000, type=int, dest='save_freq',
                        help='The save frequency, default is 10000')
    parser.add_argument('--valid_freq', action='store', default=500, type=int, dest='valid_freq',
                        help='The valid frequency, default is 500')
    parser.add_argument('--finish_after', action='store', default=10000000, type=int, dest='finish_after',
                        help='Finish after this many updates, default is 10000000')
    parser.add_argument('--discount_lr_freq', action='store', default=10000, type=int, dest='discount_lr_freq',
                        help='The discount learning rate frequency, default is 2000')
    parser.add_argument('--fix_embedding', action='store_false', default=True, dest='fix_embedding',
                        help='Fix the source embedding, default to True, set to False')
    parser.add_argument('--debug', action='store_true', default=False, dest='debug',
                        help='Open debug mode, default is False, set to True')
    parser.add_argument('--dump_hidden', action='store', default=None, dest='dump_hidden',
                        help='Dump hidden state output to file (only available in debug mode), default is None')
    parser.add_argument('--conn', action='store', default=1, type=int, dest='connection_type',
                        help='Connection type, default is 1 (divided bidirectional GRU);\n'
                             '2 is bidirectional only in first layer, other layers are forward')
    parser.add_argument('--decay_c', action="store", metavar="decay_c", dest="decay_c", type=float, default=0.0,
                        help='The L2 regularization rate, default is 0.0.')
    parser.add_argument('--clip_c', action="store", metavar="clip_c", dest="clip_c", type=float, default=1.0,
                        help='The gradient clipping rate, default is 1.0.')

    args = parser.parse_args()

    print('Arguments:')
    print(args)

    build_regression(args, dict(
        saveto='',
        preload='model/en2fr.iter160000.npz',
        reload_=True,
        dim_word=620,
        dim=1000,
        decay_c=0.,
        clip_c=1.,
        lrate=0.8,
        optimizer='sgd',
        patience=1000,
        maxlen=1000,
        batch_size=80,
        dispFreq=2500,
        saveFreq=10000,
        datasets=['./data/train/filtered_en-fr.en',
                  './data/train/filtered_en-fr.fr'],
        use_dropout=False,
        overwrite=False,
        n_words=30000,
        n_words_src=30000,
        sort_by_len=False,

        # Options from v-yanfa
        convert_embedding=False,
        dump_before_train=False,
        plot_graph=None,

        # [NOTE]: This is for old model, settings for new model are in args
        n_encoder_layers=1,
        n_decoder_layers=1,
    ))


if __name__ == '__main__':
    main()
