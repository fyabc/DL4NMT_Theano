#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
import cPickle as pkl

import numpy as np
import theano

from ..models.deliberation import DelibNMT
from ..utility.utils import prepare_data, load_params
from ..constants import profile


def _load_one_file(filename, dic, maxlen, voc_size, UNKID=1):
    ret = []
    with open(filename, 'r') as f:
        for line in f:
            X = line.strip().split()[:maxlen]
            Y = [dic.get(w, UNKID) for w in X]
            Z = [w if w < voc_size else UNKID for w in Y]
            ret.append(Z)
    return ret


def valid(modelpath,
          valid_batch_size=80,
          valid_datasets=('./data/dev/dev_en.tok',
                          './data/dev/dev_fr.tok'),
          dictionary='',
          dictionary_target='',
          start_idx=1, step_idx=1, end_idx=1,
          logfile='.log'
          ):
    model_options = pkl.load(open(modelpath + '.pkl', 'rb'))

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
    m_block = (len(valid_src) + valid_batch_size - 1) // valid_batch_size

    print 'Building model'
    model = DelibNMT(model_options)
    params = model.initializer.init_delib_params()
    model.init_tparams(params)
    # Build model
    trng, use_noise, x, x_mask, y, y_mask, y_pos_, cost, probs = model.build_model()
    use_noise.set_value(0.)
    inps = [x, x_mask, y, y_mask, y_pos_]
    print 'Build predictor'
    f_predictor = theano.function(inps, [cost, probs], profile=profile)
    print 'Done'

    print_samples = True
    fp_log = open(logfile, 'w')
    for curidx in xrange(start_idx, end_idx + 1, step_idx):
        params = load_params(os.path.splitext(modelpath)[0] + '.iter' + str(curidx) + '.npz', params)
        for (kk, vv) in params.iteritems():
            model.P[kk].set_value(vv)
        all_sample = [0 for _ in xrange(model_options['maxlen'] + 2)]
        correct_sample = [0 for _ in xrange(model_options['maxlen'] + 2)]
        for block_id in xrange(m_block):
            seqx = valid_src[block_id * valid_batch_size: (block_id + 1) * valid_batch_size]
            seqy = valid_trg[block_id * valid_batch_size: (block_id + 1) * valid_batch_size]
            x, x_mask, y, y_mask = prepare_data(seqx, seqy)
            y_pos_ = np.repeat(np.arange(y.shape[0])[:, None], y.shape[1], axis=1).astype('int64')
            cost, probs = f_predictor(x, x_mask, y, y_mask, y_pos_)
            _predict = probs.argmax(axis=1).reshape((y.shape[0], y.shape[1]))
            results = ((_predict == y).astype('float32') * y_mask).sum(axis=1)
            m_sample_ = y_mask.sum(axis=1)

            for ii, (_xx, _yy) in enumerate(zip(results, m_sample_)):
                all_sample[ii] += _yy
                correct_sample[ii] += _xx

        if print_samples:
            print all_sample
            print >> fp_log, all_sample
            print_samples = False
        for (xx_, yy_) in zip(correct_sample, all_sample):
            if yy_ < 1e-3:
                yy_ = 1.
            print xx_ / yy_, '\t',
            print >> fp_log, xx_ / yy_, '\t',
        print ''
        print >> fp_log, ''
    fp_log.close()
