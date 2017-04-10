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


def gen_sample(tparams, f_init, f_next, x, options, trng=None, k=1, maxlen=30,
               stochastic=True, argmax=False):
    """Generate sample, either with stochastic sampling or beam search. Note that,

    this function iteratively calls f_init and f_next functions.
    """

    # k is the beam size we have
    if k > 1:
        assert not stochastic, 'Beam search does not support stochastic sampling'

    sample = []
    sample_score = []
    if stochastic:
        sample_score = 0

    live_k = 1
    dead_k = 0

    hyp_samples = [[]] * live_k
    hyp_scores = np.zeros(live_k).astype(fX)
    hyp_states = []

    # get initial state of decoder rnn and encoder context
    ret = f_init(x)
    next_state, ctx0 = ret[0], ret[1]
    next_w = -1 * np.ones((1,)).astype('int64')  # bos indicator

    for ii in xrange(maxlen):
        ctx = np.tile(ctx0, [live_k, 1])
        inps = [next_w, ctx, next_state]
        ret = f_next(*inps)
        next_p, next_w, next_state = ret[0], ret[1], ret[2]

        if stochastic:
            if argmax:
                nw = next_p[0].argmax()
            else:
                nw = next_w[0]
            sample.append(nw)
            sample_score -= np.log(next_p[0, nw])
            if nw == 0:
                break
        else:
            cand_scores = hyp_scores[:, None] - np.log(next_p)
            cand_flat = cand_scores.flatten()
            ranks_flat = cand_flat.argsort()[:(k - dead_k)]

            voc_size = next_p.shape[1]
            trans_indices = ranks_flat / voc_size
            word_indices = ranks_flat % voc_size
            costs = cand_flat[ranks_flat]

            new_hyp_samples = []
            new_hyp_scores = np.zeros(k - dead_k).astype(fX)
            new_hyp_states = []

            for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                new_hyp_samples.append(hyp_samples[ti] + [wi])
                new_hyp_scores[idx] = copy.copy(costs[idx])
                new_hyp_states.append(copy.copy(next_state[ti]))

            # check the finished samples
            new_live_k = 0
            hyp_samples = []
            hyp_scores = []
            hyp_states = []

            for idx in xrange(len(new_hyp_samples)):
                if new_hyp_samples[idx][-1] == 0:
                    sample.append(new_hyp_samples[idx])
                    sample_score.append(new_hyp_scores[idx])
                    dead_k += 1
                else:
                    new_live_k += 1
                    hyp_samples.append(new_hyp_samples[idx])
                    hyp_scores.append(new_hyp_scores[idx])
                    hyp_states.append(new_hyp_states[idx])
            hyp_scores = np.array(hyp_scores)
            live_k = new_live_k

            if new_live_k < 1:
                break
            if dead_k >= k:
                break

            next_w = np.array([w[-1] for w in hyp_samples])
            next_state = np.array(hyp_states)

    if not stochastic:
        # dump every remaining one
        if live_k > 0:
            for idx in xrange(live_k):
                sample.append(hyp_samples[idx])
                sample_score.append(hyp_scores[idx])

    return sample, sample_score


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
          saveto='model.npz',
          saveFreq=1000,  # save the parameters after every saveFreq updates
          validFreq=10000,
          datasets=('/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.en.tok',
                    '/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.fr.tok'),
          picked_train_idxes_file=r'',
          use_dropout=False,
          reload_=False,
          overwrite=False,
          preload='',
          sort_by_len=False,

          # Options below are from v-yanfa
          convert_embedding=True,
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
          ):

    # Model options: load and save
    model_options = locals().copy()
    print 'Top options: '
    pprint(model_options)
    print 'Done'

    load_options(model_options, reload_, preload)

    print 'Loading data'
    text_iterator = TextIterator(
        datasets[0],
        datasets[1],
        vocab_filenames[0],
        vocab_filenames[1],
        batch_size,
        maxlen,
        n_words_src,
        n_words,
    )

    print 'Building model'
    model = NMTModel(model_options)
    params = model.initializer.init_params()

    # Reload parameters
    if reload_ and os.path.exists(preload):
        print 'Reloading model parameters'
        params = load_params(preload, params)

        # Only convert parameters when reloading
        if convert_embedding:
            # =================
            # Convert input and output embedding parameters with a exist word embedding
            # =================
            print 'Convert input and output embedding'

            temp_Wemb = params['Wemb']
            orig_emb_mean = np.mean(temp_Wemb, axis=0)

            params['Wemb'] = np.tile(orig_emb_mean, [params['Wemb'].shape[0], 1])

            # Load vocabulary map dicts and do mapping
            with open(map_filename, 'rb') as map_file:
                map_en = pkl.load(map_file)
                map_fr = pkl.load(map_file)

            for full, top in map_en.iteritems():
                emb_size = temp_Wemb.shape[0]
                if full < emb_size and top < emb_size:
                    params['Wemb'][top] = temp_Wemb[full]

            print 'Convert input embedding done'

            temp_ff_logit_W = params['ff_logit_W']
            temp_Wemb_dec = params['Wemb_dec']
            temp_b = params['ff_logit_b']

            orig_ff_logit_W_mean = np.mean(temp_ff_logit_W, axis=1)
            orig_Wemb_dec_mean = np.mean(temp_Wemb_dec, axis=0)
            orig_b_mean = np.mean(temp_b)

            params['ff_logit_W'] = np.tile(orig_ff_logit_W_mean, [params['ff_logit_W'].shape[1], 1]).T
            params['ff_logit_b'].fill(orig_b_mean)
            params['Wemb_dec'] = np.tile(orig_Wemb_dec_mean, [params['Wemb_dec'].shape[0], 1])

            for full, top in map_en.iteritems():
                emb_size = temp_Wemb.shape[0]
                if full < emb_size and top < emb_size:
                    params['ff_logit_W'][:, top] = temp_ff_logit_W[:, full]
                    params['ff_logit_b'][top] = temp_b[full]
                    params['Wemb_dec'][top] = temp_Wemb[full]

            print 'Convert output embedding done'

            # ================
            # End Convert
            # ================

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

    grads, _ = apply_gradient_clipping(clip_c, grads)

    # compile the optimizer, the actual computational graph is compiled here
    lr = tensor.scalar(name='lr')
    print 'Building optimizers...',
    f_grad_shared, f_update = eval(optimizer)(lr, model.P, grads, inps, cost)
    print 'Done'

    print 'Optimization'

    best_p = None
    bad_counter = 0
    uidx = search_start_uidx(reload_, preload)
    if reload_:
        lrate *= 0.5
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
            cost = f_grad_shared(x, x_mask, y, y_mask)

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
                print 'Epoch {} Update {} Cost {:.6f} UD {:.6f} Time {:.6f}'.format(
                    eidx, uidx, float(cost), ud, time.time() - start_time,
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
            # generate some samples with the model and display them

            if np.mod(uidx, validFreq) == 0:
                # todo: validation
                pass

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
