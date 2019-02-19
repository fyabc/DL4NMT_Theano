#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Generate candidate vocabulary on test dataset using deliberation model.

Result data: numpy zip file
Data format:
    numpy.array
    shape = (#sentences, maxlen * k)
    dtype = int64
    value = top-k word indices
"""

import cPickle as pkl
import os

import numpy as np
import theano

from ..models import build_and_init_model
from ..utility.train import get_train_input
from ..utility.basic import arg_top_k
from ..constants import Datasets, profile


def _load_one_file(filename, dic, maxlen, voc_size, UNKID=1):
    ret = []
    with open(filename, 'r') as f:
        for line in f:
            X = line.strip().split()[:maxlen]
            Y = [dic.get(w, UNKID) for w in X]
            Z = [w if w < voc_size else UNKID for w in Y]
            ret.append(Z)
    return ret


def prepare_gen(model_options):
    test_datasets = [
        os.path.join('data', 'test', Datasets[model_options['task']][-4]),
        os.path.join('data', 'test', Datasets[model_options['task']][-3]),
    ]
    dictionary, dictionary_target = model_options['vocab_filenames']

    # load valid data; truncated by the ``maxlen''
    srcdict = pkl.load(open(dictionary, 'rb'))
    trgdict = pkl.load(open(dictionary_target, 'rb'))

    kw_ret = {}

    test_src = _load_one_file(test_datasets[0], srcdict, model_options['maxlen'], model_options['n_words_src'])
    test_trg = _load_one_file(test_datasets[1], trgdict, model_options['maxlen'], model_options['n_words'])

    return test_src, test_trg, kw_ret


def generate(model_path, dump_path, k=100, test_batch_size=80):
    """Generate per-word vocabulary.

    Parameters
    ----------
    model_path
    dump_path
    k
    test_batch_size

    Returns
    -------

    """

    model, model_options, ret = build_and_init_model(model_path, model_type='DelibNMT')
    assert model_options['use_delib'], 'Must use deliberation model.'

    _, use_noise, x, x_mask, y, y_mask, y_pos_, _, cost, _, probs = ret
    use_noise.set_value(0.)
    inps = [x, x_mask, y, y_mask, y_pos_]
    print 'Build predictor'
    # todo: Need fix here, remove y inputs.
    f_predictor = theano.function(inps, [cost, probs], profile=profile)
    print 'Done'

    test_src, test_trg, kw_ret = prepare_gen(model_options)

    maxlen = model_options['maxlen']
    m_block = (len(test_src) + test_batch_size - 1) // test_batch_size

    print 'Test data size: {}, maxlen: {}, m_block: {}'.format(len(test_src), maxlen, m_block)

    result = np.zeros([len(test_src), (maxlen + 1) * k], dtype='int64')

    for block_id in xrange(m_block):
        seqx = test_src[block_id * test_batch_size: (block_id + 1) * test_batch_size]
        seqy = test_trg[block_id * test_batch_size: (block_id + 1) * test_batch_size]

        inputs = get_train_input(seqx, seqy, maxlen=None, use_delib=True)
        x, x_mask = inputs[0], inputs[1]
        cost, probs = f_predictor(*inputs)

        _predict = arg_top_k(-probs, k, axis=1)[:, :k]

        print '$ x: {}, x_mask: {}, _predict: {}'.format(x.shape, x_mask.shape, _predict.shape)

        _predict = _predict.reshape([-1, x.shape[1], k])
        _min_len = min(x_mask.shape[0], _predict.shape[0])
        x_mask_i = np.zeros(_predict.shape[:2], dtype='int64')
        x_mask_i[:_min_len] = x_mask[:_min_len]
        _predict *= x_mask_i[:, :, None]
        _predict = np.transpose(_predict, [1, 0, 2]).reshape([x.shape[1], -1])

        result[block_id * test_batch_size: block_id * test_batch_size + x.shape[1], :_predict.shape[1]] = _predict

    np.savez(dump_path, delib_vocab=result)
    print 'Result dump to {}.'.format(dump_path)