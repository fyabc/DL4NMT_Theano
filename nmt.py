"""
Build a neural machine translation model with soft attention
"""

import cPickle as pkl
import copy
import os
import sys
import time
from pprint import pprint

import numpy as np
import theano
import theano.tensor as tensor

from constants import profile, fX
from data_iterator import TextIterator
from optimizers import *
from utils import *
from model import NMTModel


def pred_probs(f_log_probs, prepare_data, options, iterator, verbose=True, normalize=False):
    """Calculate the log probablities on a given corpus using translation model"""

    probs = []

    n_done = 0

    for x, y in iterator:
        n_done += len(x)

        lengths = np.array([len(s) for s in x])

        x, x_mask, y, y_mask = prepare_data(x, y)

        pprobs = f_log_probs(x, x_mask, y, y_mask)
        if normalize:
            pprobs = pprobs / lengths

        for pp in pprobs:
            probs.append(pp)

        sys.stdout.write('\rDid ' + str(n_done) + ' samples')

    print
    return np.array(probs)


def validation(iterator, f_cost, maxlen=None):
    valid_cost = 0.0
    valid_count = 0
    for x, y in iterator:
        x, x_mask, y, y_mask = prepare_data(x, y, maxlen=maxlen)

        if x is None:
            continue

        valid_cost += f_cost(x, x_mask, y, y_mask)
        valid_count += 1

    return valid_cost / valid_count


def _just_ref():
    """Just reference something to prevent them from being optimized out by PyCharm."""

    _ = sgd, adadelta, rmsprop, adam


def train(dim_word=100,  # word vector dimensionality
          dim=1000,  # the number of LSTM units
          encoder='gru',
          decoder='gru_cond',
          n_words_src=30000,
          n_words=30000,
          patience=10,  # early stopping patience
          max_epochs=5000,
          finish_after=10000000,  # finish after this many updates
          dispFreq=100,
          decay_c=0.,  # L2 regularization penalty
          alpha_c=0.,  # alignment regularization
          clip_c=-1.,  # gradient clipping threshold
          lrate=1.,  # learning rate
          maxlen=100,  # maximum length of the description
          optimizer='rmsprop',
          batch_size=16,
          valid_batch_size=80,
          saveto='model.npz',
          saveFreq=1000,  # save the parameters after every saveFreq updates
          validFreq=2500,
          datasets=('/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.en.tok',
                    '/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.fr.tok'),
          valid_datasets=('./data/dev/dev_en.tok',
                          './data/dev/dev_fr.tok'),
          small_train_datasets=('./data/train/small_en-fr.en',
                                './data/train/small_en-fr.fr'),
          use_dropout=False,
          reload_=False,
          overwrite=False,
          preload='',

          # Options below are from v-yanfa
          dump_before_train=False,
          plot_graph=None,
          vocab_filenames=('./data/dic/filtered_dic_en-fr.en.pkl',
                           './data/dic/filtered_dic_en-fr.fr.pkl'),
          map_filename='./data/dic/mapFullVocab2Top1MVocab.pkl',
          lr_discount_freq=80000,

          # Options of deeper encoder and decoder
          n_encoder_layers=1,
          n_decoder_layers=1,
          encoder_many_bidirectional=True,

          attention_layer_id=0,
          unit='gru',
          residual_enc=None,
          residual_dec=None,
          use_zigzag=False,

          initializer='orthogonal',
          given_embedding=None,
          ):

    # Model options: load and save
    model_options = locals().copy()
    print 'Top options: '
    pprint(model_options)
    print 'Done'

    load_options(model_options, reload_, preload)

    print 'Loading data'
    text_iterator = TextIterator(
        datasets[0], datasets[1],
        vocab_filenames[0], vocab_filenames[1],
        batch_size, maxlen, n_words_src, n_words,
    )

    valid_iterator = TextIterator(
        valid_datasets[0], valid_datasets[1],
        vocab_filenames[0], vocab_filenames[1],
        valid_batch_size, maxlen, n_words_src, n_words,
    )

    small_train_iterator = TextIterator(
        small_train_datasets[0], small_train_datasets[1],
        vocab_filenames[0], vocab_filenames[1],
        batch_size, maxlen, n_words_src, n_words,
    )

    print 'Building model'
    model = NMTModel(model_options)
    params = model.initializer.init_params()

    # Reload parameters
    if reload_ and os.path.exists(preload):
        print 'Reloading model parameters'
        load_params(preload, params)

    # Given embedding
    if given_embedding is not None:
        print 'Loading given embedding...',
        load_embedding(params, given_embedding)
        print 'Done'

    if True:
        print_params(params)

    model.init_tparams(params)

    # Build model
    trng, use_noise, \
        x, x_mask, y, y_mask, \
        opt_ret, \
        cost, x_emb = model.build_model()
    inps = [x, x_mask, y, y_mask]

    print 'Building sampler'
    f_init, f_next = model.build_sampler(trng=trng, use_noise=use_noise)

    # before any regularizer
    print 'Building f_log_probs...',
    f_log_probs = theano.function(inps, cost, profile=profile)
    f_x_emb = theano.function([x, x_mask], x_emb, profile=profile)
    print 'Done'
    sys.stdout.flush()
    cost = cost.mean()

    cost = l2_regularization(cost, model.P, decay_c)

    cost = regularize_alpha_weights(cost, alpha_c, model_options, x_mask, y_mask, opt_ret)

    print 'Building f_cost...',
    f_cost = theano.function(inps, cost, profile=profile)
    print 'Done'

    if plot_graph is not None:
        print 'Plotting post-compile graph...',
        theano.printing.pydotprint(
            f_cost,
            outfile='pictures/post_compile_{}'.format(plot_graph),
            var_with_name_simple=True,
        )
        print 'Done'

    print 'Computing gradient...',
    grads = tensor.grad(cost, wrt=itemlist(model.P))
    print 'Done'
    sys.stdout.flush()

    grads, g2 = apply_gradient_clipping(clip_c, grads)

    # compile the optimizer, the actual computational graph is compiled here
    lr = tensor.scalar(name='lr')
    print 'Building optimizers...',
    f_grad_shared, f_update = eval(optimizer)(lr, model.P, grads, inps, cost, g2=g2)
    print 'Done'

    print 'Optimization'

    best_p = None
    bad_counter = 0
    uidx = search_start_uidx(reload_, preload)
    print 'uidx', uidx, 'l_rate', lrate

    estop = False
    history_errs = []

    if dump_before_train:
        print 'Dumping before train...',
        saveto_uidx = '{}.iter{}.npz'.format(
            os.path.splitext(saveto)[0], uidx)
        np.savez(saveto_uidx, history_errs=history_errs,
                 uidx=uidx, **unzip(model.P))
        save_options(model_options, uidx, saveto)
        print 'Done'

    start_time = time.time()

    for eidx in xrange(max_epochs):
        n_samples = 0

        for i, (x, y) in enumerate(text_iterator):
            n_samples += len(x)
            uidx += 1
            use_noise.set_value(1.)

            x, x_mask, y, y_mask = prepare_data(x, y, maxlen=maxlen)

            if x is None:
                print 'Minibatch with zero sample under length ', maxlen
                uidx -= 1
                continue

            ud_start = time.time()

            # compute cost, grads and copy grads to shared variables
            cost, g2_value = f_grad_shared(x, x_mask, y, y_mask)

            # do the update on parameters
            f_update(lrate)

            ud = time.time() - ud_start

            # check for bad numbers, usually we remove non-finite elements
            # and continue training - but not done here
            if np.isnan(cost) or np.isinf(cost):
                print 'NaN detected'
                return 1., 1., 1.

            # discount reward
            if lr_discount_freq > 0 and np.mod(uidx, lr_discount_freq) == 0:
                lrate *= 0.5
                print 'Discount learning rate to {} at iteration {}'.format(lrate, uidx)

            # verbose
            if np.mod(uidx, dispFreq) == 0:
                print 'Epoch {} Update {} Cost {:.5f} G2 {:.5f} UD {:.5f} Time {:.5f} s'.format(
                    eidx, uidx, float(cost), float(g2_value), ud, time.time() - start_time,
                )
                sys.stdout.flush()

            if np.mod(uidx, saveFreq) == 0:
                # save with uidx
                if not overwrite:
                    print 'Saving the model at iteration {}...'.format(uidx),
                    saveto_uidx = '{}.iter{}.npz'.format(
                        os.path.splitext(saveto)[0], uidx)
                    np.savez(saveto_uidx, history_errs=history_errs,
                             uidx=uidx, **unzip(model.P))
                    save_options(model_options, uidx, saveto)
                    print 'Done'
                    sys.stdout.flush()

            if np.mod(uidx, validFreq) == 0:
                valid_cost = validation(valid_iterator, f_cost, maxlen=maxlen)
                small_train_cost = validation(small_train_iterator, f_cost, maxlen=maxlen)
                print 'Valid cost {:.5f} Small train cost {:.5f}'.format(valid_cost, small_train_cost)
                sys.stdout.flush()

            # finish after this many updates
            if uidx >= finish_after:
                print 'Finishing after {} iterations!'.format(uidx)
                estop = True
                break

        print 'Seen {} samples'.format(n_samples)

        if estop:
            break

    if best_p is not None:
        zipp(best_p, model.P)

    use_noise.set_value(0.)

    return 0.


if __name__ == '__main__':
    pass
