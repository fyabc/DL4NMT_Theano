#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import cPickle as pkl

import numpy as np
import theano

try:
    from bottleneck import argpartsort as part_sort
except ImportError:
    from bottleneck import argpartition as part_sort

from ..models.deliberation import DelibNMT
from ..utility.utils import prepare_data, load_params, set_logging_file, message, get_logging_file, load_options_test
from ..constants import profile, Datasets


def _load_one_file(filename, dic, maxlen, voc_size, UNKID=1):
    ret = []
    with open(filename, 'r') as f:
        for line in f:
            X = line.strip().split()[:maxlen]
            Y = [dic.get(w, UNKID) for w in X]
            Z = [w if w < voc_size else UNKID for w in Y]
            ret.append(Z)
    return ret


def prepare_predict(modelpath,
                    valid_datasets=('./data/dev/dev_en.tok',
                                    './data/dev/dev_fr.tok'),
                    dictionary='',
                    dictionary_target='',
                    logfile='.log'):
    model_options = load_options_test(modelpath)

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

    return model_options, model, valid_src, valid_trg, params, f_predictor


def predict(modelpath,
            action='a',
            valid_batch_size=80,
            valid_datasets=('./data/dev/dev_en.tok',
                            './data/dev/dev_fr.tok'),
            dictionary='',
            dictionary_target='',
            start_idx=1, step_idx=1, end_idx=1,
            logfile='.log',
            k=1,
            ):
    model_options, model, valid_src, valid_trg, params, f_predictor = prepare_predict(
        modelpath, valid_datasets, dictionary, dictionary_target, logfile,
    )

    m_block = (len(valid_src) + valid_batch_size - 1) // valid_batch_size
    print_samples = True

    for curidx in xrange(start_idx, end_idx + 1, step_idx):
        params = load_params(os.path.splitext(os.path.splitext(modelpath)[0])[0] +
                             '.iter' + str(curidx) + '.npz', params)
        for (kk, vv) in params.iteritems():
            model.P[kk].set_value(vv)
        all_sample = [0 for _ in xrange(model_options['maxlen'] + 2)]
        correct_sample = [0 for _ in xrange(model_options['maxlen'] + 2)]
        all_precisions = []
        all_recalls = []

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
                _predict = part_sort(probs, k, axis=1)

                y_mask_i = y_mask.astype('int64')

                for s_idx in xrange(y.shape[1]):
                    # Words of the sentence
                    # [NOTE]: EOS = 0
                    R = set((y * y_mask_i)[:, s_idx].flatten())

                    # Words of top-k prediction of the sentence
                    s_predict = _predict.reshape((y.shape[0], y.shape[1], _predict.shape[-1]))[:, s_idx, :]
                    T_n = set((s_predict * y_mask_i[:, s_idx]).flatten())

                    if 'p' in action:
                        all_precisions.append(len(R.intersection(T_n)) * 1.0 / len(T_n))
                    if 'r' in action:
                        all_recalls.append(len(R.intersection(T_n)) * 1.0 / len(R))
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
            message('Precision: top {} = {}'.format(k, np.mean(all_precisions)))
        if 'r' in action:
            message('Recall: top {} = {}'.format(k, np.mean(all_recalls)))

        message()

    get_logging_file().close()
