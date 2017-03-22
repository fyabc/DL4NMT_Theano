"""
Build a neural machine translation model with soft attention
"""

import cPickle as pkl
import copy
import os
import re
import sys
import time

import numpy as np
import theano
import theano.tensor as tensor

from constants import profile, fX
from data_iterator import TextIterator
from layers import init_params, build_model, build_sampler
from optimizers import *
from utils import load_params, init_tparams, zipp, unzip, itemlist, prepare_data


def gen_sample(tparams, f_init, f_next, x, options, trng=None, k=1, maxlen=30,
               stochastic=True, argmax=False):
    """Generate sample, either with stochastic sampling or beam search. Note that,

    this function iteratively calls f_init and f_next functions.
    """

    # k is the beam size we have
    if k > 1:
        assert not stochastic, \
            'Beam search does not support stochastic sampling'

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


def train(dim_word=100,             # word vector dimensionality
          dim=1000,                 # the number of LSTM units
          encoder='gru',
          decoder='gru_cond',
          n_words_src=30000,
          n_words=30000,
          patience=10,              # early stopping patience
          max_epochs=5000,
          finish_after=10000000,    # finish after this many updates
          dispFreq=100,
          decay_c=0.,               # L2 regularization penalty
          alpha_c=0.,               # alignment regularization
          clip_c=-1.,               # gradient clipping threshold
          lrate=1.,                 # learning rate
          maxlen=100,               # maximum length of the description
          optimizer='rmsprop',
          batch_size=16,
          saveto='model.npz',
          saveFreq=1000,            # save the parameters after every saveFreq updates
          datasets=('/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.en.tok',
                    '/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.fr.tok'),
          picked_train_idxes_file=r'',
          use_dropout=False,
          reload_=False,
          overwrite=False,
          preload='',
          sort_by_len=False,

          # Options from v-yanfa
          convert_embedding=True,
          dump_before_train=False,
          plot_graph=None,
          vocab_filenames=('./data/dic/filtered_dic_en-fr.en.pkl',
                           './data/dic/filtered_dic_en-fr.fr.pkl'),
          map_filename='./data/dic/mapFullVocab2Top1MVocab.pkl',
          lr_discount_freq=80000,
          ):
    # Model options
    model_options = locals().copy()
    if reload_:
        lrate *= 0.5

    # load dictionaries and invert them

    # reload options
    if reload_ and os.path.exists(preload):
        print 'Reloading model options'
        with open(r'.\model\en2fr.iter160000.npz.pkl', 'rb') as f:
            # model_options = pkl.load(f)
            # FIXME: Update the option instead of replace it
            model_options.update(pkl.load(f))

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

    # sys.stdout.flush()
    # train_data_x = pkl.load(open(datasets[0], 'rb'))
    # train_data_y = pkl.load(open(datasets[1], 'rb'))
    #
    # if len(picked_train_idxes_file) != 0:
    #     picked_idxes = pkl.load(open(picked_train_idxes_file, 'rb'))
    #     train_data_x = [train_data_x[id] for id in picked_idxes]
    #     train_data_y = [train_data_y[id] for id in picked_idxes]
    #
    # print 'Total train:', len(train_data_x)
    # print 'Max len:', max([len(x) for x in train_data_x])
    # sys.stdout.flush()
    #
    # if sort_by_len:
    #     slen = np.array([len(s) for s in train_data_x])
    #     sidx = slen.argsort()
    #
    #     _sbuf = [train_data_x[i] for i in sidx]
    #     _tbuf = [train_data_y[i] for i in sidx]
    #
    #     train_data_x = _sbuf
    #     train_data_y = _tbuf
    #     print len(train_data_x[0]), len(train_data_x[-1])
    #     sys.stdout.flush()
    #     train_batch_idx = get_minibatches_idx(len(train_data_x), batch_size, shuffle=False)
    # else:
    #     train_batch_idx = get_minibatches_idx(len(train_data_x), batch_size, shuffle=True)

    print 'Building model'
    params = init_params(model_options)
    # reload parameters
    if reload_ and os.path.exists(preload):
        print 'Reloading model parameters'
        params = load_params(preload, params)

        # for k, v in params.iteritems():
        #     print '>', k, v.shape, v.dtype

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

            # for k, v in params.iteritems():
            #     print '>', k, v.shape, v.dtype

            # ================
            # End Convert
            # ================

    tparams = init_tparams(params)

    trng, use_noise, \
        x, x_mask, y, y_mask, \
        opt_ret, \
        cost, x_emb = \
        build_model(tparams, model_options)
    inps = [x, x_mask, y, y_mask]

    print 'Building sampler'
    f_init, f_next = build_sampler(tparams, model_options, trng, use_noise)

    # before any regularizer
    print 'Building f_log_probs...',
    f_log_probs = theano.function(inps, cost, profile=profile)
    f_x_emb = theano.function([x, x_mask], x_emb, profile=profile)
    print 'Done'
    sys.stdout.flush()
    cost = cost.mean()

    # apply L2 regularization on weights
    if decay_c > 0.:
        decay_c = theano.shared(np.float32(decay_c), name='decay_c')
        weight_decay = 0.
        for kk, vv in tparams.iteritems():
            weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    # regularize the alpha weights
    if alpha_c > 0. and not model_options['decoder'].endswith('simple'):
        alpha_c = theano.shared(np.float32(alpha_c), name='alpha_c')
        alpha_reg = alpha_c * (
            (tensor.cast(y_mask.sum(0) // x_mask.sum(0), fX)[:, None] -
             opt_ret['dec_alphas'].sum(0)) ** 2).sum(1).mean()
        cost += alpha_reg

    # after all regularizers - compile the computational graph for cost
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
    grads = tensor.grad(cost, wrt=itemlist(tparams))
    print 'Done'
    sys.stdout.flush()
    # apply gradient clipping here
    if clip_c > 0.:
        g2 = 0.
        for g in grads:
            g2 += (g ** 2).sum()
        new_grads = []
        for g in grads:
            new_grads.append(tensor.switch(g2 > (clip_c ** 2),
                                           g / tensor.sqrt(g2) * clip_c,
                                           g))
        grads = new_grads

    # compile the optimizer, the actual computational graph is compiled here
    lr = tensor.scalar(name='lr')
    print 'Building optimizers...',
    f_grad_shared, f_update = eval(optimizer)(lr, tparams, grads, inps, cost)
    print 'Done'

    print 'Optimization'

    best_p = None
    bad_counter = 0
    uidx = 0
    if reload_:
        m = re.search('.+iter(\d+?)\.npz', preload)
        if m:
            uidx = int(m.group(1))
    print 'uidx', uidx, 'l_rate', lrate

    estop = False
    history_errs = []
    # reload history

    if dump_before_train:
        print 'Dumping before train...',
        saveto_uidx = '{}.iter{}.npz'.format(
            os.path.splitext(saveto)[0], uidx)
        np.savez(saveto_uidx, history_errs=history_errs,
                 uidx=uidx, **unzip(tparams))
        print 'Done'

    if saveFreq == -1:
        saveFreq = len(train[0]) / batch_size

    for eidx in xrange(max_epochs):
        n_samples = 0

        # for i, batch_idx in train_batch_idx:
        #
        #     x = [train_data_x[id] for id in batch_idx]
        #     y = [train_data_y[id] for id in batch_idx]

        for i, (x, y) in enumerate(text_iterator):
            n_samples += len(x)
            uidx += 1
            use_noise.set_value(1.)

            x, x_mask, y, y_mask = prepare_data(x, y)

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
                print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost, 'UD ', ud
                sys.stdout.flush()

            if np.mod(uidx, saveFreq) == 0:
                # save with uidx
                if not overwrite:
                    # print 'Saving the model at iteration {}...'.format(uidx),
                    saveto_uidx = '{}.iter{}.npz'.format(
                        os.path.splitext(saveto)[0], uidx)
                    np.savez(saveto_uidx, history_errs=history_errs,
                             uidx=uidx, **unzip(tparams))
                    # print 'Done'
                    # sys.stdout.flush()
            # generate some samples with the model and display them

            # finish after this many updates
            if uidx >= finish_after:
                print 'Finishing after %d iterations!' % uidx
                estop = True
                break

        print 'Seen %d samples' % n_samples

        if estop:
            break

    if best_p is not None:
        zipp(best_p, tparams)

    use_noise.set_value(0.)

    return 0.


if __name__ == '__main__':
    pass
