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
from theano.updates import OrderedUpdates

from constants import profile, fX, boundary_dict
from data_iterator import TextIterator
from optimizers import Optimizers
from utils import *

from utils_fine_tune import *
from model import NMTModel

from hmrnn import prepare_explicit_boundary

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


def validation(iterator, f_cost, encoder_unit, decoder_unit,
               use_enc_explicit_boundary=False,
               use_dec_explicit_boundary=False,
               maxlen=None,
               source_idict=None, target_idict=None,
               f_get_boundary=None,
               encoder_boundary_save_file=None, decoder_boundary_save_file=None,
               encoder_boundary_before_sigmoid_save_file=None,
               encoder_r_boundary_before_sigmoid_save_file=None,
               decoder_boundary_before_sigmoid_save_file=None,
               data_file=None):

    valid_cost = 0.0
    valid_count = 0

    encoder_per = 0.0
    decoder_per = 0.0

    x_expected_error_num_0 = 0.0
    x_expected_error_tot_0 = 0.0
    x_expected_error_num_1 = 0.0
    x_expected_error_tot_1 = 0.0

    y_expected_error_num_0 = 0.0
    y_expected_error_tot_0 = 0.0
    y_expected_error_num_1 = 0.0
    y_expected_error_tot_1 = 0.0

    for iters in iterator:

        x, y = iters[:2]
        iters = iters[2:]

        if 'hmrnn' in encoder_unit and use_enc_explicit_boundary:
            explicit_boundary_x = iters[0]
            iters = iters[1:]

        # if 'hmrnn' in decoder_unit and use_dec_explicit_boundary:
        #     explicit_boundary_y = iters[0]
        #     iters = iters[1:]

        # print '\n'
        # print '\n'.join([' '.join([source_idict[w] for w in line]) for line in x])

        assert len(iters) == 0

        ret = prepare_data(x, y, maxlen=maxlen, explicit_boundary_x=explicit_boundary_x)

        x, x_mask, y, y_mask = ret[:4]
        ret = ret[4:]

        if 'hmrnn' in encoder_unit and use_enc_explicit_boundary:
            explicit_boundary_x = ret[0]
            ret = ret[1:]


        assert len(ret) == 0

        if x is None:
            continue
        inps = [x, x_mask, y, y_mask]

        if 'hmrnn' in encoder_unit and use_enc_explicit_boundary:
            inps.append(explicit_boundary_x)

        if 'hmrnn' in decoder_unit and use_dec_explicit_boundary:
            inps.append(prepare_explicit_boundary(y, target_idict))
        #FIXME: Fix NaN (explicit_boundary)
        ret = f_cost(*inps)

        batch_valid_cost = ret[0]
        ret = ret[1:]
        valid_cost += batch_valid_cost

        if 'hmrnn' in encoder_unit:
            batch_encoder_per, encoder_boundary = ret[:2]
            ret = ret[2:]

            encoder_per += batch_encoder_per.mean()

            encoder_boundary = encoder_boundary.reshape((encoder_boundary.shape[0], encoder_boundary.shape[1]))
            x_expected_boundary_mask_0 = np.zeros_like(x)
            x_expected_boundary_mask_1 = np.zeros_like(x)

            for i in xrange(x.shape[0]):
                for j in xrange(x.shape[1]):
                    if source_idict[x[i][j]][-2:] == '@@':
                        x_expected_boundary_mask_0[i][j] = 1.0
                    elif i > 0 and source_idict[x[i - 1][j]][-2:] == '@@' and source_idict[x[i][j]][-2:] != '@@':
                        x_expected_boundary_mask_1[i][j] = 1.0

            x_expected_error_num_0 += (encoder_boundary * x_expected_boundary_mask_0).sum()
            x_expected_error_tot_0 += x_expected_boundary_mask_0.sum()

            x_expected_error_num_1 += ((1.0 - encoder_boundary) * x_expected_boundary_mask_1).sum()
            x_expected_error_tot_1 += x_expected_boundary_mask_1.sum()

        if 'hmrnn' in decoder_unit:
            batch_decoder_per, decoder_boundary = ret[:2]
            ret = ret[2:]

            decoder_per += batch_decoder_per.mean()

            decoder_boundary = decoder_boundary.reshape((decoder_boundary.shape[0], decoder_boundary.shape[1]))
            y_expected_boundary_mask_0 = np.zeros_like(y)
            y_expected_boundary_mask_1 = np.zeros_like(y)

            for i in xrange(y.shape[0]):
                for j in xrange(y.shape[1]):
                    if i > 0 and target_idict[y[i - 1][j]][-2:] == '@@':
                        y_expected_boundary_mask_0[i][j] = 1.0
                    elif i > 1 and target_idict[y[i - 2][j]][-2:] == '@@' and target_idict[y[i - 1][j]][-2:] != '@@':
                        y_expected_boundary_mask_1[i][j] = 1.0

            y_expected_error_num_0 += (decoder_boundary * y_expected_boundary_mask_0).sum()
            y_expected_error_tot_0 += y_expected_boundary_mask_0.sum()

            y_expected_error_num_1 += ((1.0 - decoder_boundary) * y_expected_boundary_mask_1).sum()
            y_expected_error_tot_1 += y_expected_boundary_mask_1.sum()

        assert ret == []

        if valid_count == 0 and f_get_boundary is not None:

            ret = f_get_boundary(*inps)

            if encoder_boundary_save_file is not None:
                enc_b, enc_b_bs = ret[:2]
                ret = ret[2:]

                enc_b = np.reshape(enc_b, (enc_b.shape[0], enc_b.shape[1])).T
                np.savetxt(encoder_boundary_save_file, enc_b, delimiter=', ', fmt='%.2f')
                encoder_boundary_save_file.write('\n')
                encoder_boundary_save_file.flush()

                enc_b_bs = np.reshape(enc_b_bs, (enc_b_bs.shape[0], enc_b_bs.shape[1])).T
                np.savetxt(encoder_boundary_before_sigmoid_save_file, enc_b_bs, delimiter=', ', fmt='%.5f')
                encoder_boundary_before_sigmoid_save_file.write('\n')
                encoder_boundary_before_sigmoid_save_file.flush()

                if encoder_r_boundary_before_sigmoid_save_file is not None:
                    enc_r_b_bs = ret[0]
                    ret = ret[1:]

                    enc_r_b_bs = np.reshape(enc_r_b_bs, (enc_r_b_bs.shape[0], enc_r_b_bs.shape[1])).T
                    np.savetxt(encoder_r_boundary_before_sigmoid_save_file, enc_r_b_bs, delimiter=', ', fmt='%.5f')
                    encoder_r_boundary_before_sigmoid_save_file.write('\n')
                    encoder_r_boundary_before_sigmoid_save_file.flush()

            if decoder_boundary_save_file is not None:
                dec_b, dec_b_bs = ret[:2]
                ret = ret[2:]

                dec_b = np.reshape(dec_b, (dec_b.shape[0], dec_b.shape[1])).T
                np.savetxt(decoder_boundary_save_file, dec_b, delimiter=', ', fmt='%.2f')
                decoder_boundary_save_file.write('\n')
                decoder_boundary_save_file.flush()

                dec_b_bs = np.reshape(dec_b_bs, (dec_b_bs.shape[0], dec_b_bs.shape[1])).T
                np.savetxt(decoder_boundary_before_sigmoid_save_file, dec_b_bs, delimiter=', ', fmt='%.5f')
                decoder_boundary_before_sigmoid_save_file.write('\n')
                decoder_boundary_before_sigmoid_save_file.flush()

            assert ret == []

            assert data_file is not None, 'data_file must be provided'

            reshaped_x = np.reshape(x, (x.shape[0], x.shape[1])).T
            np.savetxt(data_file, reshaped_x, delimiter=', ', fmt='%6d')
            data_file.write('\n')

            for line in reshaped_x:
                str_line = []
                for word in line:
                    str_word = source_idict[word]
                    if str_word == ',':
                        str_word = '<comma>'
                    if str_word == '.':
                        str_word = '<period>'
                    str_line.append(str_word)
                data_file.write(', '.join(str_line))
                data_file.write('\n')
            data_file.write('\n')

            np.savetxt(data_file, np.reshape(x_mask, (x_mask.shape[0], x_mask.shape[1])).T, delimiter=', ', fmt='%.2f')
            data_file.write('\n')

            reshaped_y = np.reshape(y, (y.shape[0], y.shape[1])).T
            np.savetxt(data_file, reshaped_y, delimiter=', ', fmt='%6d')
            data_file.write('\n')

            for line in reshaped_y:
                str_line = []
                for word in line:
                    str_word = target_idict[word]
                    if str_word == ',':
                        str_word = '<comma>'
                    if str_word == '.':
                        str_word = '<period>'
                    str_line.append(str_word)
                data_file.write(', '.join(str_line))
                data_file.write('\n')
            data_file.write('\n')

            np.savetxt(data_file, np.reshape(y_mask, (y_mask.shape[0], y_mask.shape[1])).T, delimiter=', ', fmt='%.2f')
            data_file.write('\n')
            data_file.flush()

        valid_count += 1

    if x_expected_error_tot_0 == 0.0:
        x_expected_error_tot_0 = 1.0
    if x_expected_error_tot_1 == 0.0:
        x_expected_error_tot_1 = 1.0
    if y_expected_error_tot_0 == 0.0:
        y_expected_error_tot_0 = 1.0
    if y_expected_error_tot_1 == 0.0:
        y_expected_error_tot_1 = 1.0

    return (
        valid_cost / valid_count,
        encoder_per / valid_count,
        decoder_per / valid_count,
        1.0 - x_expected_error_num_0 / x_expected_error_tot_0,
        1.0 - x_expected_error_num_1 / x_expected_error_tot_1,
        1.0 - y_expected_error_num_0 / y_expected_error_tot_0,
        1.0 - y_expected_error_num_1 / y_expected_error_tot_1,
    )


def train(dim_word=100,  # word vector dimensionality
          dim=1000,  # the number of LSTM units
          # encoder='gru',
          # decoder='gru_cond',
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
          bleuFreq=5000,
          bleu_start_id=0,
          datasets=('/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.en.tok',
                    '/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.fr.tok'),
          valid_datasets=('./data/dev/dev_en.tok',
                          './data/dev/dev_fr.tok'),
          test_datasets=('./data/test/test_en-fr.en.tok',
                         './data/test/test_en-fr.fr.tok'),
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
          encoder_unit=None,
          decoder_unit=None,

          residual_enc=None,
          residual_dec=None,
          use_zigzag=False,

          initializer='orthogonal',
          given_embedding=None,

          dist_type=None,
          sync_batch=0,
          dist_recover_lr_iter=False,
          sync_models=False,

          unit_size=2,
          cond_unit_size=2,

          given_imm=False,
          dump_imm=False,
          shuffle_data=False,

          decoder_all_attention=False,
          average_context=False,
          task='en-fr',

          fine_tune_patience=8,

          # for hmrnn

          bottom_lstm=False,
          use_mask=False,
          boundary_type_str='ST',
          hard_sigmoid_a_schedule=5.0,
          temperature_schedule=1.0,
          benefit_0_boundary=False,
          use_enc_explicit_boundary=False,
          use_dec_explicit_boundary=False,
          use_all_one_boundary=False,
          enc_boundary_regularization=0.0,
          dec_boundary_regularization=0.0,
          ):
    model_options = locals().copy()

    # Set distributed computing environment
    worker_id = 0
    if dist_type == 'mv':
        try:
            import multiverso as mv
        except ImportError:
            import multiverso_ as mv

        worker_id = mv.worker_id()
    elif dist_type == 'mpi_reduce' :
        from mpi4py import MPI
        mpi_communicator = MPI.COMM_WORLD
        worker_id = mpi_communicator.Get_rank()
        workers_cnt = mpi_communicator.Get_size()
    if dist_type and not sync_models:
        sync_batch = 1 #force sync gradient in every mini-batch

    print 'Use {}, worker id: {}'.format('multiverso' if dist_type == 'mv' else 'mpi' if dist_recover_lr_iter else 'none', worker_id)
    sys.stdout.flush()

    current_time_str = time.strftime('%m-%d-%H-%M-%S')

    print 'Logging file name:', \
        'log/complete/e{}d{}_res{}_att{}_worker{}_task{}_{}.txt'.format(
            n_encoder_layers, n_decoder_layers, residual_enc, attention_layer_id,
            worker_id, task, current_time_str,
        )
    # Set logging file
    set_logging_file('log/complete/e{}d{}_res{}_att{}_worker{}_task{}_{}.txt'.format(
        n_encoder_layers, n_decoder_layers, residual_enc, attention_layer_id,
        worker_id, task, current_time_str,
    ))
    if 'hmrnn' in encoder_unit:
        encoder_boundary_save_file = open('log/complete/boundaries/boundary_encoder_{}.csv'.format(current_time_str), 'w')
        encoder_boundary_before_sigmoid_save_file = open('log/complete/boundaries/boundary_encoder_before_sigmoid_{}.csv'.format(current_time_str), 'w')
        encoder_r_boundary_before_sigmoid_save_file = \
            None if bottom_lstm else open('log/complete/boundaries/boundary_encoder_r_before_sigmoid_{}.csv'.format(current_time_str), 'w')
    else:
        encoder_boundary_save_file = None
        encoder_boundary_before_sigmoid_save_file = None
        encoder_r_boundary_before_sigmoid_save_file = None

    if 'hmrnn' in decoder_unit:
        decoder_boundary_save_file = open('log/complete/boundaries/boundary_decoder_{}.csv'.format(current_time_str), 'w')
        decoder_boundary_before_sigmoid_save_file = open('log/complete/boundaries/boundary_decoder_before_sigmoid_{}.csv'.format(current_time_str), 'w')
    else:
        decoder_boundary_save_file = None
        decoder_boundary_before_sigmoid_save_file = None

    data_file = open('log/complete/boundaries/data_{}.csv'.format(current_time_str), 'w')

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

    load_options(model_options, reload_, preload)
    check_options(model_options)

    print 'Loading data'
    log('\n\n\nStart to prepare data\n@Current Time = {}'.format(time.time()))
    sys.stdout.flush()

    if dist_type:
        dataset_src = '{}_{}'.format(datasets[0], worker_id)
        dataset_tgt = '{}_{}'.format(datasets[1], worker_id)
    else:
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
            enc_explicit_boundary= dataset_src + '.boundary' if use_enc_explicit_boundary else None,
            # dec_explicit_boundary= dataset_tgt + '.boundary' if use_dec_explicit_boundary else None,
        )

    valid_iterator = TextIterator(
        valid_datasets[0], valid_datasets[1],
        vocab_filenames[0], vocab_filenames[1],
        valid_batch_size, maxlen, n_words_src, n_words,
        enc_explicit_boundary= valid_datasets[0] + '.boundary' if use_enc_explicit_boundary else None,
        # dec_explicit_boundary= valid_datasets[1] + '.boundary' if use_dec_explicit_boundary else None,
    )

    small_train_iterator = TextIterator(
        small_train_datasets[0], small_train_datasets[1],
        vocab_filenames[0], vocab_filenames[1],
        batch_size, maxlen, n_words_src, n_words,
        enc_explicit_boundary= small_train_datasets[0] + '.boundary' if use_enc_explicit_boundary else None,
        # dec_explicit_boundary= small_train_datasets[1] + '.boundary' if use_dec_explicit_boundary else None,
        # print_data_file=open('log/complete/data_raw_{}.txt'.format(current_time_str), 'w')
    )

    test_iterator = TextIterator(
        test_datasets[0], test_datasets[1],
        vocab_filenames[0], vocab_filenames[1],
        valid_batch_size, maxlen, n_words_src, n_words,
        enc_explicit_boundary= test_datasets[0] + '.boundary' if use_enc_explicit_boundary else None,
        # dec_explicit_boundary= test_datasets[1] + '.boundary' if use_dec_explicit_boundary else None,
    )

    with open(vocab_filenames[0], 'rb') as f:
        source_dict = pkl.load(f)
        source_idict = {v: k for k, v in source_dict.iteritems()}
        source_idict[0] = '<eos>'
        source_idict[1] = 'UNK'

    with open(vocab_filenames[1], 'rb') as f:
        target_dict = pkl.load(f)
        target_idict = {v: k for k, v in target_dict.iteritems()}
        target_idict[0] = '<eos>'
        target_idict[1] = 'UNK'

    print 'Building model'
    model = NMTModel(model_options)
    params = model.initializer.init_params()

    # Reload parameters
    if reload_ and os.path.exists(preload):
        print 'Reloading model parameters'
        load_params(preload, params)
    sys.stdout.flush()

    # Given embedding
    if given_embedding is not None:
        print 'Loading given embedding...',
        load_embedding(params, given_embedding)
        print 'Done'

    print_params(params)

    model.init_tparams(params)

    # Build model
    trng, use_noise, boundary_type, hard_sigmoid_a, temperature,\
        x, x_mask, y, y_mask, \
        explicit_boundary_x, explicit_boundary_y, opt_ret, \
        cost, x_emb, \
        encoder_boundary, decoder_boundary,\
        stochastic_updates = model.build_model()

    boundary_type.set_value(boundary_dict[boundary_type_str])
    hard_sigmoid_a.set_value(hard_sigmoid_a_schedule)


    if 'hmrnn' in encoder_unit:
        encoder_boundary_before_sigmoid = model.encoder_boundary_before_sigmoid
        if not bottom_lstm:
            encoder_r_boundary_before_sigmoid = model.encoder_r_boundary_before_sigmoid
        encoder_boundary_percent = (encoder_boundary * x_mask[:, :, None]).sum(0).flatten() / x_mask.sum(0)

    if 'hmrnn' in decoder_unit:
        decoder_boundary_before_sigmoid = model.decoder_boundary_before_sigmoid
        decoder_boundary_percent = (decoder_boundary * y_mask[:, :, None]).sum(0).flatten() / y_mask.sum(0)

    all_stochastic_updates = OrderedUpdates()
    for updates in stochastic_updates:
        all_stochastic_updates.update(updates)

    inps = [x, x_mask, y, y_mask]

    if 'hmrnn' in encoder_unit and use_enc_explicit_boundary:
        inps.append(explicit_boundary_x)

    if 'hmrnn' in decoder_unit and use_dec_explicit_boundary:
        inps.append(explicit_boundary_y)

    print 'Building f_get_boundary...',
    get_boundary_outputs = []

    if 'hmrnn' in encoder_unit:
        get_boundary_outputs.append(encoder_boundary)
        get_boundary_outputs.append(encoder_boundary_before_sigmoid)
        if not bottom_lstm:
            get_boundary_outputs.append(encoder_r_boundary_before_sigmoid)

    if 'hmrnn' in decoder_unit:
        get_boundary_outputs.append(decoder_boundary)
        get_boundary_outputs.append(decoder_boundary_before_sigmoid)

    f_get_boundary = theano.function(
        inps,
        get_boundary_outputs,
        profile=profile, updates=all_stochastic_updates) \
        if get_boundary_outputs else None

    print 'Done'

    print 'Building sampler'
    # f_init, f_next = model.build_sampler(trng=trng, use_noise=use_noise)
    f_init, f_next = model.build_sampler(trng=trng, use_noise=use_noise, batch_mode=True,
                                         dropout=model_options['use_dropout'], boundary_type=boundary_type,
                                         use_enc_explicit_boundary=model_options['use_enc_explicit_boundary'],
                                         use_dec_explicit_boundary=model_options['use_dec_explicit_boundary'],
                                         hard_sigmoid_a=hard_sigmoid_a, temperature=temperature)

    # before any regularizer
    print 'Building f_log_probs...',
    f_log_probs = theano.function(inps, cost, profile=profile, updates=all_stochastic_updates)
    # f_x_emb = theano.function([x, x_mask], x_emb, profile=profile, updates=all_stochastic_updates)
    print 'Done'


    sys.stdout.flush()
    cost = cost.mean()

    cost = l2_regularization(cost, model.P, decay_c)

    cost = regularize_alpha_weights(cost, alpha_c, model_options, x_mask, y_mask, opt_ret)

    decoder_boundary_regularization_cost = None

    if enc_boundary_regularization > 0.0:
        if 'hmrnn' in encoder_unit:
            encoder_boundary_regularization_cost = \
                enc_boundary_regularization * ((encoder_boundary * x_mask[:, :, None])**2).mean()
            cost += encoder_boundary_regularization_cost

    if dec_boundary_regularization > 0.0:
        if 'hmrnn' in decoder_unit:
            decoder_boundary_regularization_cost = \
                dec_boundary_regularization * ((decoder_boundary * y_mask[:, :, None])**2).mean()
            cost += decoder_boundary_regularization_cost

    print 'Building f_cost...',
    cost_outputs = [cost]
    if 'hmrnn' in encoder_unit:
        cost_outputs.append(encoder_boundary_percent)
        cost_outputs.append(encoder_boundary)
    if 'hmrnn' in decoder_unit:
        cost_outputs.append(decoder_boundary_percent)
        cost_outputs.append(decoder_boundary)
    f_cost = theano.function(inps, cost_outputs, profile=profile, updates=all_stochastic_updates)
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
    grads = tensor.grad(cost, wrt=itemlist(model.P), disconnected_inputs='warn')

    clip_shared = theano.shared(np.array(clip_c, dtype=fX), name='clip_shared')
    grads, g2 = clip_grad_remove_nan(grads, clip_shared, model.P)

    # compile the optimizer, the actual computational graph is compiled here
    lr = tensor.scalar(name='lr')
    print 'Building optimizers...',

    given_imm_data = get_adadelta_imm_data(optimizer, given_imm, saveto)

    f_grad_shared, f_update, grads_shared, imm_shared = Optimizers[optimizer](
        lr, model.P, grads, inps, cost, g2=g2, given_imm_data=given_imm_data, dump_imm=dump_imm,
        stochastic_updates=all_stochastic_updates, extra_costs=decoder_boundary_regularization_cost)
    print 'Done'

    print 'Optimization'
    log('Preparation Done\n@Current Time = {}'.format(time.time()))

    if dist_type == 'mv':
        mv.barrier()
    elif dist_type == 'mpi_reduce':
        mpi_communicator.Barrier()
        #create receive buffers for mpi allreduce
        if sync_models:
            rec_models = [np.zeros_like(p.get_value()) for p in model.P.itervalues()]
            rec_immes_list = [None for _ in imm_shared]
            for i in xrange(len(imm_shared)):
                rec_immes_list[i] = [np.zeros_like(p.get_value()) for p in model.P.itervalues()]
        else:
            rec_grads = [np.zeros_like(p.get_value()) for p in model.P.itervalues()]

    estop = False
    history_errs = []
    best_bleu = -1.0
    best_p = None
    bad_counter = 0
    uidx = search_start_uidx(reload_, preload)
    print 'uidx', uidx, 'l_rate', lrate
    start_uidx = uidx

    if dump_before_train:
        print 'Dumping before train...',
        saveto_uidx = '{}.iter{}.npz'.format(
            os.path.splitext(saveto)[0], uidx)
        np.savez(saveto_uidx, history_errs=history_errs,
                 uidx=uidx, **unzip(model.P))
        save_options(model_options, uidx, saveto)
        print 'Done'

    commu_time_sum = 0.0
    cp_time_sum =0.0
    reduce_time_sum=0.0

    start_time = time.time()

    def print_cost():
        use_noise.set_value(0.)
        small_train_cost, \
            small_train_enc_per, small_train_dec_per, \
            small_train_enc_exp_0, small_train_enc_exp_1, \
            small_train_dec_exp_0, small_train_dec_exp_1 = \
            validation(small_train_iterator, f_cost, maxlen=maxlen,
                       encoder_unit=encoder_unit, decoder_unit=decoder_unit,
                       use_enc_explicit_boundary=use_enc_explicit_boundary,
                       use_dec_explicit_boundary=use_dec_explicit_boundary,
                       source_idict=source_idict, target_idict=target_idict,
                       f_get_boundary=f_get_boundary,
                       encoder_boundary_save_file=encoder_boundary_save_file,
                       decoder_boundary_save_file=decoder_boundary_save_file,
                       encoder_boundary_before_sigmoid_save_file=encoder_boundary_before_sigmoid_save_file,
                       encoder_r_boundary_before_sigmoid_save_file=encoder_r_boundary_before_sigmoid_save_file,
                       decoder_boundary_before_sigmoid_save_file=decoder_boundary_before_sigmoid_save_file,
                       data_file=data_file)

        message('Small train cost {:.5f}'.format(small_train_cost))
        message('Small train boundary percentage: encoder {:.3f}, decoder {:.3f}'
                .format(small_train_enc_per, small_train_dec_per))
        message('Small train expected boundary percentage: encoder {:.3f}-{:.3f}, decoder {:.3f}-{:.3f}'
                .format(small_train_enc_exp_0, small_train_enc_exp_1,
                        small_train_dec_exp_0, small_train_dec_exp_1))

        valid_cost, \
            valid_enc_per, valid_dec_per, \
            valid_enc_exp_0, valid_enc_exp_1, \
            valid_dec_exp_0, valid_dec_exp_1 = \
            validation(valid_iterator, f_cost, encoder_unit=encoder_unit, decoder_unit=decoder_unit,
                       use_enc_explicit_boundary=use_enc_explicit_boundary,
                       use_dec_explicit_boundary=use_dec_explicit_boundary,
                       source_idict=source_idict, target_idict=target_idict, maxlen=maxlen)
        message('Valid cost {:.5f}'.format(valid_cost))
        message('Valid boundary percentage: encoder {:.3f}, decoder {:.3f}'.format(valid_enc_per, valid_dec_per))
        message('Valid expected boundary percentage: encoder {:.3f}-{:.3f}, decoder {:.3f}-{:.3f}'
                .format(valid_enc_exp_0, valid_enc_exp_1, valid_dec_exp_0, valid_dec_exp_1))

        test_cost, \
            test_enc_per, test_dec_per, \
            test_enc_exp_0, test_enc_exp_1, \
            test_dec_exp_0, test_dec_exp_1 = \
            validation(test_iterator, f_cost, encoder_unit=encoder_unit, decoder_unit=decoder_unit,
                       use_enc_explicit_boundary=use_enc_explicit_boundary,
                       use_dec_explicit_boundary=use_dec_explicit_boundary,
                       source_idict=source_idict, target_idict=target_idict, maxlen=maxlen)
        message('Test cost {:.5f}'.format(test_cost))
        message('Test boundary percentage: encoder {:.3f}, decoder {:.3f}'.format(test_enc_per, test_dec_per))
        message('Test expected boundary percentage: encoder {:.3f}-{:.3f}, decoder {:.3f}-{:.3f}'
                .format(test_enc_exp_0, test_enc_exp_1, test_dec_exp_0, test_dec_exp_1))

        ST_test_cost = test_cost

        if ('hmrnn' in encoder_unit or 'hmrnn' in decoder_unit) and boundary_type_str != 'ST':
            boundary_type.set_value(boundary_dict['ST'])

            ST_test_cost, \
            ST_test_enc_per, ST_test_dec_per, \
            ST_test_enc_exp_0, ST_test_enc_exp_1, \
            ST_test_dec_exp_0, ST_test_dec_exp_1 = \
                validation(test_iterator, f_cost, encoder_unit=encoder_unit, decoder_unit=decoder_unit,
                           use_enc_explicit_boundary=use_enc_explicit_boundary,
                           use_dec_explicit_boundary=use_dec_explicit_boundary,
                           source_idict=source_idict, target_idict=target_idict, maxlen=maxlen)
            message('ST Test cost {:.5f}'.format(ST_test_cost))
            message(
                'ST Test boundary percentage: encoder {:.3f}, decoder {:.3f}'.format(ST_test_enc_per, ST_test_dec_per))
            message('ST Test expected boundary percentage: encoder {:.3f}-{:.3f}, decoder {:.3f}-{:.3f}'
                    .format(ST_test_enc_exp_0, ST_test_enc_exp_1, ST_test_dec_exp_0, ST_test_dec_exp_1))

            boundary_type.set_value(boundary_dict[boundary_type_str])

        sys.stdout.flush()
        message('Bias of boundary:', model.P['decoder_b'].get_value()[:, 4*dim])
        return small_train_cost, valid_cost, test_cost, ST_test_cost

    print_cost()

    for eidx in xrange(max_epochs):
        if shuffle_data:
            text_iterator = load_shuffle_text_iterator(
                eidx, text_iterator_list,
                datasets, vocab_filenames, batch_size, maxlen, n_words_src, n_words,
                use_enc_explicit_boundary=use_enc_explicit_boundary,
                use_dec_explicit_boundary=use_dec_explicit_boundary,
            )

        n_samples = 0
        if dist_type == 'mpi_reduce':
            mpi_communicator.Barrier()

        for i, iters in enumerate(text_iterator):
            x, y = iters[:2]
            iters = iters[2:]

            if 'hmrnn' in encoder_unit and use_enc_explicit_boundary:
                explicit_boundary_x = iters[0]
                iters = iters[1:]

            # if 'hmrnn' in decoder_unit and use_dec_explicit_boundary:
            #     explicit_boundary_y = iters[0]
            #     iters = iters[1:]

            assert len(iters) == 0

            n_samples += len(x)
            uidx += 1
            use_noise.set_value(1.)

            #TODO: prepare boundaries while prepare data

            ret = prepare_data(x, y, maxlen=maxlen, explicit_boundary_x=explicit_boundary_x)

            x, x_mask, y, y_mask = ret[:4]
            ret = ret[4:]

            if 'hmrnn' in encoder_unit and use_enc_explicit_boundary:
                explicit_boundary_x = ret[0]
                ret = ret[1:]

            assert len(ret) == 0

            inps = [x, x_mask, y, y_mask]
            if 'hmrnn' in encoder_unit and use_enc_explicit_boundary:
                inps.append(explicit_boundary_x)
            if 'hmrnn' in decoder_unit and use_dec_explicit_boundary:
                inps.append(prepare_explicit_boundary(y, target_idict))

            if x is None:
                print 'Minibatch with zero sample under length ', maxlen
                uidx -= 1
                continue

            effective_uidx = uidx - start_uidx
            ud_start = time.time()

            # compute cost, grads and copy grads to shared variables
            cost, g2_value = f_grad_shared(*inps)

            # print 'Compute cost finished'

            if dist_type == 'mpi_reduce' and uidx % sync_batch == 0:
                reduce_start = time.time()
                commu_time = 0

                gpucpu_cp_time = 0
                if sync_models: #sync model parameters
                    commu_time_delta, cp_time_delta = all_reduce_params(model.P.itervalues(), rec_models, workers_cnt)
                    commu_time += commu_time_delta
                    gpucpu_cp_time += cp_time_delta
                    for i in xrange(len(imm_shared)):  # sync immediate parameters in ada* algorithm
                        commu_time_delta, cp_time_delta = all_reduce_params(imm_shared[i], rec_immes_list[i], workers_cnt)
                        commu_time += commu_time_delta
                        gpucpu_cp_time += cp_time_delta
                else: #sync gradients
                    commu_time, gpucpu_cp_time = all_reduce_params(grads_shared, rec_grads)

                reduce_time = time.time() - reduce_start
                commu_time_sum += commu_time
                reduce_time_sum += reduce_time
                cp_time_sum += gpucpu_cp_time

                print '@Worker = {}, Reduce time = {:.5f}, Commu time = {:.5f}, GPUCPU time = {:.5f}'.format(
                    worker_id, reduce_time_sum / effective_uidx, commu_time_sum / effective_uidx, cp_time_sum /effective_uidx)

            # do the update on parameters

            curr_lr = lrate if not dist_type or dist_recover_lr_iter < effective_uidx else lrate * 0.05 + effective_uidx * lrate / dist_recover_lr_iter * 0.95
            if curr_lr < lrate:
                print 'Curr lr %.3f' % curr_lr

            f_update(curr_lr)

            ud = time.time() - ud_start

            # check for bad numbers, usually we remove non-finite elements
            # and continue training - but not done here
            if np.isnan(cost) or np.isinf(cost):
                message('NaN detected')
                sys.stdout.flush()
                return 1., 1., 1.

            # discount reward
            # FIXME: Do NOT enable this and fine-tune at the same time
            if lr_discount_freq > 0 and np.mod(uidx, lr_discount_freq) == 0:
                lrate *= 0.5
                clip_shared.set_value(clip_shared.get_value() * 0.5)
                message('Discount learning rate to {} at iteration {}'.format(lrate, uidx))

            # sync batch
            if dist_type == 'mv' and np.mod(uidx, dispFreq) == 0:
                comm_start = time.time()
                model.sync_tparams()
                message('@Comm time = {:.5f}'.format(time.time() - comm_start))

            # verbose
            if np.mod(uidx, dispFreq) == 0:
                message('Epoch {} Update {} Cost {:.5f} G2 {:.5f} UD {:.5f} Time {:.5f} s'.format(
                    eidx, uidx, float(cost), float(g2_value), ud, time.time() - start_time,
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
                small_train_cost, valid_cost, test_cost, ST_test_cost = print_cost()

            # Fine-tune based on dev BLEU
            if bleuFreq > 0 and np.mod(uidx, bleuFreq) == 0 and uidx >= bleu_start_id:

                valid_bleu = translate_dev_get_bleu(
                    model, f_init, f_next, trng, use_noise
                )
                message('BLEU Valid = {:.2f} at iteration {}'.format(valid_bleu, uidx))
                sys.stdout.flush()

                test_bleu = translate_test_get_bleu(model, f_init, f_next, trng, use_noise)
                message('BLEU Test = {:.2f} at iteration {}'.format(test_bleu, uidx))
                sys.stdout.flush()

                if ('hmrnn' in encoder_unit or 'hmrnn' in decoder_unit) and boundary_type_str != 'ST':
                    boundary_type.set_value(boundary_dict['ST'])

                    ST_test_bleu = translate_test_get_bleu(model, f_init, f_next, trng, use_noise)
                    message('BLEU ST Test = {:.2f} at iteration {}'.format(ST_test_bleu, uidx))

                    boundary_type.set_value(boundary_dict[boundary_type_str])
                    sys.stdout.flush()

                if fine_tune_patience > 0:
                    new_bleu = test_bleu

                    if new_bleu > best_bleu:
                        bad_counter = 0
                        best_bleu = new_bleu
                    else:
                        bad_counter += 1

                        if bad_counter >= fine_tune_patience:
                            print 'Fine tune:',
                            lrate *= 0.5
                            clip_shared.set_value(clip_shared.get_value() * 0.5)
                            message('Discount learning rate to {} at iteration {}'.format(lrate, uidx))
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
