"""
Build a neural machine translation model with soft attention
"""

import cPickle as pkl
import copy
import os
import sys
import time
import math
from pprint import pprint

import numpy as np
import theano
import theano.tensor as tensor

from .constants import profile, fX
from .utility.data_iterator import TextIterator
from .utility.optimizers import Optimizers
from .utility.utils import *

from .utility.translate import translate_dev_get_bleu
from .model import NMTModel


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
          dump_before_train=True,
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

          dist_type=None,
          dist_recover_lr_iter=False,
          clip_grads_local = False,

          unit_size=2,
          cond_unit_size=2,

          given_imm=False,
          dump_imm=False,
          shuffle_data=False,

          decoder_all_attention=False,
          average_context=False,
          task='en-fr',

          fine_tune_patience=8,
          nccl = False,
          src_vocab_map_file = None,
          tgt_vocab_map_file = None,
          ):
    model_options = locals().copy()

    # Set distributed computing environment
    worker_id = 0
    if dist_type == 'mv':
        try:
            import multiverso as mv
        except ImportError:
            from . import multiverso_ as mv

        worker_id = mv.worker_id()
    elif dist_type == 'mpi_reduce':
        from mpi4py import MPI
        mpi_communicator = MPI.COMM_WORLD
        worker_id = mpi_communicator.Get_rank()
        workers_cnt = mpi_communicator.Get_size()

        if nccl:
            nccl_comm = init_nccl_env(mpi_communicator)

    print 'Use {}, worker id: {}'.format('multiverso' if dist_type == 'mv' else 'mpi' if dist_recover_lr_iter else 'none', worker_id)
    sys.stdout.flush()

    # Set logging file
    set_logging_file('log/complete/e{}d{}_res{}_att{}_worker{}_task{}_{}.txt'.format(
        n_encoder_layers, n_decoder_layers, residual_enc, attention_layer_id,
        worker_id, task, time.strftime('%m-%d-%H-%M-%S'),
    ))

    log('''\
Start Time = {}
'''.format(
        time.strftime('%c'),
    ))

    # Model options: load and save
    message('Top options:')
    pprint(model_options)
    pprint(model_options, stream=get_logging_file())
    message('Done')
    sys.stdout.flush()

    load_options(model_options, reload_, preload, src_vocab_map_file and tgt_vocab_map_file)
    check_options(model_options)
    model_options['cost_normalization'] = 1
    ada_alpha = 0.95
    if dist_type == 'mpi_reduce':
        model_options['cost_normalization'] = workers_cnt
        lrate *= workers_cnt
        ada_alpha = pow(0.95, workers_cnt)

    if True:
        message('Model options:')
        pprint(model_options)
        pprint(model_options, stream=get_logging_file())
        message()

    print 'Loading data'
    log('\n\n\nStart to prepare data\n@Current Time = {}'.format(time.time()))
    sys.stdout.flush()

    dataset_src, dataset_tgt = datasets[0], datasets[1]

    if shuffle_data:
        text_iterator_list = [None for _ in range(10)]
        text_iterator = None
    else:
        text_iterator_list = None
        text_iterator = TextIterator(
            dataset_src, dataset_tgt,
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
        load_params(preload, params, src_map_file = src_vocab_map_file, tgt_map_file = tgt_vocab_map_file)
    sys.stdout.flush()

    # Given embedding
    if given_embedding is not None:
        print 'Loading given embedding...',
        load_embedding(params, given_embedding)
        print 'Done'

    print_params(params)

    model.init_tparams(params)

    # Build model
    trng, use_noise, \
        x, x_mask, y, y_mask, \
        opt_ret, \
        cost, test_cost, x_emb = model.build_model()
    inps = [x, x_mask, y, y_mask]

    print 'Building sampler'
    f_init, f_next = model.build_sampler(trng=trng, use_noise=use_noise, batch_mode=True)

    # before any regularizer
    print 'Building f_log_probs...',
    f_log_probs = theano.function(inps, cost, profile=profile)
    print 'Done'
    sys.stdout.flush()
    test_cost = test_cost.mean() #FIXME: do not regularize test_cost here

    cost = cost.mean()

    cost = l2_regularization(cost, model.P, decay_c)

    cost = regularize_alpha_weights(cost, alpha_c, model_options, x_mask, y_mask, opt_ret)

    print 'Building f_cost...',
    f_cost = theano.function(inps, test_cost, profile=profile)
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

    clip_shared = theano.shared(np.array(clip_c, dtype=fX), name='clip_shared')

    if dist_type != 'mpi_reduce' or clip_grads_local: #build grads clip into computational graph
        grads, g2 = clip_grad_remove_nan(grads, clip_shared, model.P)
    else: #do the grads clip after gradients aggregation
        g2 = None

    # compile the optimizer, the actual computational graph is compiled here
    lr = tensor.scalar(name='lr')
    print 'Building optimizers...',

    given_imm_data = get_adadelta_imm_data(optimizer, given_imm, preload)

    f_grad_shared, f_update, grads_shared, imm_shared = Optimizers[optimizer](
        lr, model.P, grads, inps, cost, g2=g2, given_imm_data=given_imm_data, alpha = ada_alpha)
    print 'Done'

    if dist_type == 'mpi_reduce':
        f_grads_clip = make_grads_clip_func(grads_shared = grads_shared, mt_tparams= model.P, clip_c_shared = clip_shared)

    print 'Optimization'
    log('Preparation Done\n@Current Time = {}'.format(time.time()))

    if dist_type == 'mv':
        mv.barrier()
    elif dist_type == 'mpi_reduce':
        #create receive buffers for mpi allreduce
        rec_grads = [np.zeros_like(p.get_value()) for p in model.P.itervalues()]

    estop = False
    history_errs = []
    best_bleu = -1.0
    best_p = None
    bad_counter = 0
    uidx = search_start_uidx(reload_, preload)
    '''
    if uidx != 0:
        epoch_n_batches = get_epoch_batch_cnt(dataset_src, dataset_tgt, vocab_filenames, batch_size, maxlen, n_words_src, n_words) \
            if worker_id == 0 else None
    else:
        epoch_n_batches = 1 #avoid heavy data IO

    if dist_type == 'mpi_reduce':
        epoch_n_batches = mpi_communicator.bcast(epoch_n_batches, root = 0)
    start_epoch = uidx / epoch_n_batches
    pass_batches = uidx % epoch_n_batches
    '''

    epoch_n_batches = 0
    start_epoch = 0
    pass_batches = 0

    print 'worker', worker_id, 'uidx', uidx, 'l_rate', lrate, 'n_batches', epoch_n_batches, 'start_epoch', start_epoch, 'pass_batches', pass_batches

    start_uidx = uidx

    if dump_before_train:
        print 'Dumping before train...',
        saveto_uidx = '{}.iter{}.npz'.format(
            os.path.splitext(saveto)[0], uidx)
        np.savez(saveto_uidx, history_errs=history_errs,
                 uidx=uidx, **unzip(model.P))
        save_options(model_options, uidx, saveto)
        print 'Done'
        sys.stdout.flush()
        raw_input()

    commu_time_sum = 0.0
    cp_time_sum =0.0
    reduce_time_sum = 0.0

    start_time = time.time()
    finetune_cnt = 0

    '''
    valid_cost = validation(valid_iterator, f_cost, maxlen=maxlen)
    small_train_cost = validation(small_train_iterator, f_cost, maxlen=maxlen)
    message('Initial: Valid cost {:.5f} Small train cost {:.5f}'.format(valid_cost, small_train_cost))
    sys.stdout.flush()
    '''

    for eidx in xrange(start_epoch, max_epochs):
        if shuffle_data:
            text_iterator = load_shuffle_text_iterator(
                eidx, worker_id, text_iterator_list,
                datasets, vocab_filenames, batch_size, maxlen, n_words_src, n_words,
            )
        n_samples = 0
        if dist_type == 'mpi_reduce':
            mpi_communicator.Barrier()

        for i, (x, y) in enumerate(text_iterator):
            if eidx == start_epoch and i < pass_batches: #ignore the first several batches when reload
                continue
            n_samples += len(x)
            uidx += 1
            use_noise.set_value(1.)

            x, x_mask, y, y_mask = prepare_data(x, y, maxlen=maxlen)

            if x is None:
                print 'Minibatch with zero sample under length ', maxlen
                uidx -= 1
                continue

            effective_uidx = uidx - start_uidx
            ud_start = time.time()

            # compute cost, grads
            if dist_type != 'mpi_reduce' or clip_grads_local:
                cost, g2_value = f_grad_shared(x, x_mask, y, y_mask)
            else:
                cost = f_grad_shared(x, x_mask, y, y_mask)

            if dist_type == 'mpi_reduce':
                reduce_start = time.time()
                commu_time = 0
                gpucpu_cp_time = 0

                grad_before = grads_shared[0].get_value()
                print 'Workder', worker_id, 'Before %.4f'% grad_before.sum()

                if not nccl:
                    commu_time, gpucpu_cp_time = all_reduce_params(grads_shared, rec_grads)
                else:
                    commu_time, gpucpu_cp_time = all_reduce_params_nccl(nccl_comm, grads_shared)

                grad_ = grads_shared[0].get_value()
                print 'Workder', worker_id, 'After %.5f, sum by lr %.2f is %.5f' % (grad_.sum(), lrate, grad_.sum() * lrate)

                reduce_time = time.time() - reduce_start

                if not clip_grads_local: #clip gradient after aggregation
                    g2_value = f_grads_clip()

                commu_time_sum += commu_time
                reduce_time_sum += reduce_time
                cp_time_sum += gpucpu_cp_time

                #print '@Worker = {}, Reduce time = {:.5f}, Commu time = {:.5f}, Copy time = {:.5f}'.format(worker_id, reduce_time, commu_time, gpucpu_cp_time)

            curr_lr = lrate if not dist_type or dist_recover_lr_iter < effective_uidx else lrate * 0.05 + effective_uidx * lrate / dist_recover_lr_iter * 0.95
            if curr_lr < lrate:
                print 'Curr lr %.3f' % curr_lr

            # do the update on parameters
            sum_before = model.P['Wemb'].get_value().sum()
            f_update(curr_lr)
            sum_after = model.P['Wemb'].get_value().sum()

            rg_before = (1 - ada_alpha) * grad_before * grad_before
            up_before = -np.sqrt(1e-6) / np.sqrt(rg_before + 1e-6) * grad_before

            rg = (1 - ada_alpha) * grad_ * grad_
            up = -np.sqrt(1e-6) / np.sqrt(rg + 1e-6) * grad_

            print 'Workder', worker_id, 'rg %.6f' % rg.sum(), 'delta %.6f'% (sum_after - sum_before), 'Assumed change %.6f' % (up * lrate).sum(), 'Assumed change2 %.6f' % (up_before * lrate).sum()
            sys.stdout.flush()

            raw_input()

            ud = time.time() - ud_start

            # check for bad numbers, usually we remove non-finite elements
            # and continue training - but not done here
            if np.isnan(cost) or np.isinf(cost):
                message('NaN detected')
                sys.stdout.flush()
                return 1., 1., 1.

            # discount learning rate
            # FIXME: Do NOT enable this and fine-tune at the same time
            if lr_discount_freq > 0 and np.mod(effective_uidx, lr_discount_freq) == 0:
                lrate *= 0.5
                message('Discount learning rate to {} at iteration {}'.format(lrate, uidx))

            # sync batch
            if dist_type == 'mv' and np.mod(uidx, dispFreq) == 0:
                comm_start = time.time()
                model.sync_tparams()
                message('@Comm time = {:.5f}'.format(time.time() - comm_start))

            # verbose
            if np.mod(uidx, dispFreq) == 0:
                message('Worker {} Epoch {} Update {} Cost {:.5f} G2 {:.5f} UD {:.5f} Time {:.5f} s'.format(
                    worker_id, eidx, uidx, float(cost), float(g2_value), ud, time.time() - start_time,
                ))
                sys.stdout.flush()

            if np.mod(uidx, saveFreq) == 0 and worker_id == 0:
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

                # save immediate data in adadelta
                dump_adadelta_imm_data(optimizer, imm_shared, dump_imm, saveto)

            if np.mod(uidx, validFreq) == 0:
                valid_cost = validation(valid_iterator, f_cost, maxlen=maxlen)
                small_train_cost = validation(small_train_iterator, f_cost, maxlen=maxlen)
                message('Valid cost {:.5f} Small train cost {:.5f}'.format(valid_cost, small_train_cost))
                sys.stdout.flush()

                # Fine-tune based on dev BLEU
                if fine_tune_patience > 0:
                    new_bleu = translate_dev_get_bleu(model, f_init, f_next, trng, use_noise)

                    message('BLEU = {:.2f} at iteration {}'.format(new_bleu, uidx))

                    if new_bleu > best_bleu:
                        bad_counter = 0
                        best_bleu = new_bleu
                    else:
                        bad_counter += 1
                        if bad_counter >= fine_tune_patience:
                            print 'Fine tune:',
                            if finetune_cnt % 2 == 0:
                                lrate *= 0.5
                                message('Discount learning rate to {} at iteration {}'.format(lrate, uidx))
                            else:
                                clip_shared.set_value(clip_shared.get_value() * 0.5)
                                message('Discount clip value to {} at iteration {}'.format(clip_shared.get_value(), uidx))
                            finetune_cnt += 1
                            bad_counter = 0

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
