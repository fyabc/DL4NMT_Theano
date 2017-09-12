#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
import cPickle as pkl

import numpy as np
import theano

from ..models import build_and_init_model
from ..models.deliberation import DelibNMT
from ..utility.utils import prepare_data, load_params, set_logging_file, message, load_options_test
from ..constants import profile
from ..utility.translate import translate_whole, words2seqs


def _load_one_file(filename, dic, maxlen, voc_size, UNKID=1):
    ret = []
    with open(filename, 'r') as f:
        for line in f:
            X = line.strip().split()[:maxlen]
            Y = [dic.get(w, UNKID) for w in X]
            Z = [w if w < voc_size else UNKID for w in Y]
            ret.append(Z)
    return ret


def prepare_predict(model_options,
                    valid_datasets=('./data/dev/dev_en.tok',
                                    './data/dev/dev_fr.tok'),
                    dictionary='',
                    dictionary_target='',
                    logfile='.log'):
    if not (dictionary and dictionary_target and valid_datasets):
        # Load data path from options
        valid_datasets = model_options['valid_datasets']
        dictionary, dictionary_target = model_options['vocab_filenames']

    # load valid data; truncated by the ``maxlen''
    srcdict = pkl.load(open(dictionary, 'rb'))
    trgdict = pkl.load(open(dictionary_target, 'rb'))

    def _load_files():
        src_lines = _load_one_file(valid_datasets[0], srcdict, model_options['maxlen'], model_options['n_words_src'])
        trg_lines = _load_one_file(valid_datasets[1], trgdict, model_options['maxlen'], model_options['n_words'])

        # sorted by trg file
        trgsize = [len(x) for x in trg_lines]
        sidx = np.argsort(trgsize)[::-1]
        sorted_src = [src_lines[ii] for ii in sidx]
        sorted_trg = [trg_lines[ii] for ii in sidx]
        return sorted_src, sorted_trg

    valid_src, valid_trg = _load_files()

    if logfile is not None:
        set_logging_file(logfile)

    print 'Building model'
    model = DelibNMT(model_options)
    params = model.initializer.init_params()
    model.init_tparams(params)
    # Build model
    _, use_noise, x, x_mask, y, y_mask, y_pos_, _, cost, _, probs = model.build_model()
    use_noise.set_value(0.)
    inps = [x, x_mask, y, y_mask, y_pos_]
    print 'Build predictor'
    f_predictor = theano.function(inps, [cost, probs], profile=profile)
    print 'Done'

    return model, valid_src, valid_trg, params, f_predictor


def predict(modelpath,
            action='a',
            valid_batch_size=80,
            valid_datasets=('./data/dev/dev_en.tok',
                            './data/dev/dev_fr.tok'),
            dictionary='',
            dictionary_target='',
            start_idx=1, step_idx=1, end_idx=1,
            logfile='.log',
            k_list=(1,),
            ):
    model_options = load_options_test(modelpath)

    eos_id = 0
    print_samples = True

    if model_options['use_delib']:
        model, valid_src, valid_trg, params, f_predictor = prepare_predict(
            model_options, valid_datasets, dictionary, dictionary_target, logfile,
        )

        m_block = (len(valid_src) + valid_batch_size - 1) // valid_batch_size

        for curidx in xrange(start_idx, end_idx + 1, step_idx):
            params = load_params(os.path.splitext(os.path.splitext(modelpath)[0])[0] +
                                 '.iter' + str(curidx) + '.npz', params)
            for (kk, vv) in params.iteritems():
                model.P[kk].set_value(vv)
            all_sample = [0 for _ in xrange(model_options['maxlen'] + 2)]
            correct_sample = [0 for _ in xrange(model_options['maxlen'] + 2)]
            all_precisions_list = [[] for _ in xrange(len(k_list))]
            all_recalls_list = [[] for _ in xrange(len(k_list))]

            for block_id in xrange(m_block):
                seqx = valid_src[block_id * valid_batch_size: (block_id + 1) * valid_batch_size]
                seqy = valid_trg[block_id * valid_batch_size: (block_id + 1) * valid_batch_size]
                x, x_mask, y, y_mask = prepare_data(seqx, seqy)
                y_pos_ = np.repeat(np.arange(y.shape[0])[:, None], y.shape[1], axis=1).astype('int64')
                cost, probs = f_predictor(x, x_mask, y, y_mask, y_pos_)

                if 'a' in action:
                    _predict = probs.argmax(axis=1).reshape((y.shape[0], y.shape[1]))
                    results = ((_predict == y).astype('float32') * y_mask).sum(axis=1)

                    m_sample_ = y_mask.sum(axis=1)

                    for ii, (_xx, _yy) in enumerate(zip(results, m_sample_)):
                        all_sample[ii] += _yy
                        correct_sample[ii] += _xx
                if 'p' in action or 'r' in action:
                    y_mask_i = y_mask.astype('int64')

                    for i, k in enumerate(k_list):
                        try:
                            from bottleneck import argpartsort as part_sort
                            _predict = part_sort(probs, k, axis=1)
                        except ImportError:
                            from bottleneck import argpartition as part_sort
                            _predict = part_sort(probs, k - 1, axis=1)
                        _predict = _predict.reshape((y.shape[0], y.shape[1], _predict.shape[-1]))

                        for s_idx in xrange(y.shape[1]):
                            # Words of the sentence
                            # [NOTE]: EOS = 0
                            R = set((y * y_mask_i)[:, s_idx].flatten())
                            R.discard(eos_id)

                            # Words of top-k prediction of the sentence
                            s_predict = _predict[:, s_idx, :k]
                            T_n = set((s_predict * y_mask_i[:, s_idx, None]).flatten())
                            T_n.discard(eos_id)

                            if 'p' in action:
                                all_precisions_list[i].append(len(R.intersection(T_n)) * 1.0 / len(T_n))
                            if 'r' in action:
                                all_recalls_list[i].append(len(R.intersection(T_n)) * 1.0 / len(R))
                else:
                    raise Exception('Unknown action {}'.format(action))

            message('Iteration:', curidx)

            if 'a' in action:
                message('Accuracy:')
                if print_samples:
                    message(all_sample)
                    print_samples = False
                for (xx_, yy_) in zip(correct_sample, all_sample):
                    if yy_ < 1e-3:
                        yy_ = 1.
                    message(xx_ / yy_, '\t', end='')
                message()
            if 'p' in action:
                message('Precision:\n\t{}'.format(
                    '\n\t'.join(
                        'top {} = {}'.format(k, np.mean(all_precisions))
                        for k, all_precisions in zip(k_list, all_precisions_list)
                    )))
            if 'r' in action:
                message('Recall:\n\t{}'.format(
                    '\n\t'.join(
                        'top {} = {}'.format(k, np.mean(all_recalls))
                        for k, all_recalls in zip(k_list, all_recalls_list)
                    )))

            message()
    else:
        # [NOTE]: Only use k_list[0] now.
        k = k_list[0]

        # TODO: support zh-en model.

        if not (dictionary and dictionary_target and valid_datasets):
            # Load data path from options
            valid_datasets = model_options['valid_datasets']
            dictionary, dictionary_target = model_options['vocab_filenames']

        from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
        trng = RandomStreams(1234)
        use_noise = theano.shared(np.float32(0.))

        model_type = 'NMTModel'
        if model_options['trg_attention_layer_id'] is not None:
            model_type = 'TrgAttnNMTModel'

        model, _ = build_and_init_model(modelpath, options=model_options, build=False, model_type=model_type)

        f_init, f_next = model.build_sampler(trng=trng, use_noise=use_noise, batch_mode=True,
                                             dropout=model_options['use_dropout'])

        for curidx in xrange(start_idx, end_idx + 1, step_idx):
            params = load_params(os.path.splitext(os.path.splitext(modelpath)[0])[0] +
                                 '.iter' + str(curidx) + '.npz', {})
            for (kk, vv) in params.iteritems():
                model.P[kk].set_value(vv)

            trans_str, all_cand_trans_str = translate_whole(
                model, f_init, f_next, trng, dictionary, dictionary_target, valid_datasets[0], k=k, normalize=True,
                src_trg_table=None, zhen=False, n_words_src=model_options['n_words_src'], echo=False, batch_size=32,
            )

            with open(dictionary_target, 'rb') as f:
                word_dict_trg = pkl.load(f)
            with open(valid_datasets[1], 'r') as f:
                y_seqs = words2seqs(list(f), word_dict_trg, model_options['n_words'])

            trans = words2seqs(trans_str, word_dict_trg, model_options['n_words'])
            all_cand_trans = words2seqs(all_cand_trans_str, word_dict_trg, model_options['n_words'])

            all_sample = [0 for _ in xrange(model_options['maxlen'] + 2)]
            correct_sample = [0 for _ in xrange(model_options['maxlen'] + 2)]
            all_precisions = []
            all_recalls = []

            for i, (y_seq, trans_seq) in enumerate(zip(y_seqs, trans)):
                cand_seqs = all_cand_trans[i * k: (i + 1) * k]

                R = set(y_seq)
                R.discard(eos_id)
                T_n = set()
                for cand_seq in cand_seqs:
                    T_n = T_n.union(set(cand_seq))
                T_n.discard(eos_id)

                if 'a' in action:
                    for j in xrange(len(y_seq)):
                        if j >= len(all_sample) or j >= len(correct_sample):
                            all_sample.append(0)
                            correct_sample.append(0)
                        all_sample[j] += 1
                        if len(trans_seq) > j and y_seq[j] == trans_seq[j]:
                            correct_sample[j] += 1
                if 'p' in action:
                    all_precisions.append(len(R.intersection(T_n)) * 1.0 / len(T_n))
                if 'r' in action:
                    all_recalls.append(len(R.intersection(T_n)) * 1.0 / len(R))

            message('Iteration:', curidx)

            if 'a' in action:
                message('Accuracy:')
                if print_samples:
                    message(all_sample)
                    print_samples = False
                for (xx_, yy_) in zip(correct_sample, all_sample):
                    if yy_ < 1e-3:
                        yy_ = 1.
                    message(xx_ / yy_, '\t', end='')
                message()
            if 'p' in action:
                message('Precision:\n\t{}'.format(
                    'top {} = {}'.format(k, np.mean(all_precisions))
                ))
            if 'r' in action:
                message('Recall:\n\t{}'.format(
                    'top {} = {}'.format(k, np.mean(all_recalls))
                ))

            message()
