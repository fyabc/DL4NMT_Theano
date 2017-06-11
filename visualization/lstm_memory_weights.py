#! /usr/bin/python
# -*- encoding: utf-8 -*-

"""Visualize memory weights of LSTM models."""

import sys
import os
import cPickle as pkl
from pprint import pprint
import argparse

import theano
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config import DefaultOptions
from model import NMTModel
from utils import load_params
from constants import Datasets

__author__ = 'fyabc'


def seq2words(tgt_seq, tgt_dict):
    words = []

    for w in tgt_seq:
        if w == 0:
            break
        words.append(tgt_dict[w])

    return ' '.join(words)


def build_model(model_name, options):
    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
    trng = RandomStreams(1234)
    use_noise = theano.shared(np.float32(0.))

    model = NMTModel(options)

    # allocate model parameters
    params = model.initializer.init_params()
    # load model parameters and set theano shared variables
    params = load_params(model_name, params)
    model.init_tparams(params)

    # word index
    f_init, f_next = model.build_sampler(trng=trng, use_noise=use_noise)

    return model, f_init, f_next, trng


def translate_sentence(src_seq, build_result, k, normalize):
    model, f_init, f_next, trng = build_result

    # sample given an input sequence and obtain scores
    sample, score, kw_ret = model.gen_sample(
        f_init, f_next,
        np.array(src_seq).reshape([len(src_seq), 1]),
        trng=trng, k=k, maxlen=200,
        stochastic=False, argmax=False,
        ret_memory=True,
    )

    # normalize scores according to sequence lengths
    if normalize:
        lengths = np.array([len(s) for s in sample])
        score = score / lengths
    sidx = np.argmin(score)

    return sample[sidx], kw_ret


def main(model_name, dictionary, dictionary_target, source_file, args,
         k=5, normalize=False, chr_level=False):
    # load model model_options
    with open('%s.pkl' % model_name, 'rb') as f:
        options = DefaultOptions.copy()
        options.update(pkl.load(f))

        print 'Options:'
        pprint(options)

    # load source dictionary and invert
    print 'Load and invert source dictionary...',
    with open(dictionary, 'rb') as f:
        word_dict = pkl.load(f)
    word_idict = dict()
    for kk, vv in word_dict.iteritems():
        word_idict[vv] = kk
    word_idict[0] = '<eos>'
    word_idict[1] = 'UNK'
    print 'Done'

    # load target dictionary and invert
    print 'Load and invert target dictionary...',
    with open(dictionary_target, 'rb') as f:
        word_dict_trg = pkl.load(f)
    word_idict_trg = dict()
    for kk, vv in word_dict_trg.iteritems():
        word_idict_trg[vv] = kk
    word_idict_trg[0] = '<eos>'
    word_idict_trg[1] = 'UNK'
    print 'Done'

    inputs = []
    lines = []

    print 'Loading input...',
    with open(source_file, 'r') as f:
        for idx, line in enumerate(f):
            if idx >= args.test_number:
                break

            lines.append(line)
            if chr_level:
                words = list(line.decode('utf-8').strip())
            else:
                words = line.strip().split()

            x = [word_dict[w] if w in word_dict else 1 for w in words]
            x = [ii if ii < options['n_words_src'] else 1 for ii in x]
            x.append(0)

            inputs.append(x)
    print 'Done'

    print 'Building model...',
    build_result = build_model(model_name, options)
    print 'Done'

    print '=============================='

    for i, src_seq in enumerate(inputs):
        print 'Translating sentence {}:'.format(i)
        print 'Input sentence:', lines[i],

        tgt_seq, kw_ret = translate_sentence(src_seq, build_result, k, normalize)

        print 'Output sentence:', seq2words(tgt_seq, word_dict_trg),
        print 'Visualize LSTM memory:', 'TODO'

        print


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Visualize LSTM memory weights when translating')
    parser.add_argument('-k', type=int, default=4,
                        help='Beam size (?), default is %(default)s, can also use 12')
    parser.add_argument('-p', type=int, default=5,
                        help='Number of parallel processes, default is %(default)s')
    parser.add_argument('-n', action="store_true", default=False,
                        help='Use normalize, default to False, set to True')
    parser.add_argument('-c', action="store_true", default=False,
                        help='Char level model, default to False, set to True')
    parser.add_argument('model', type=str, help='The model path')
    parser.add_argument('dictionary_source', type=str, help='The source dict path')
    parser.add_argument('dictionary_target', type=str, help='The target dict path')
    parser.add_argument('source', type=str, help='The source input path')
    parser.add_argument('-N', '--number', type=int, default=1, dest='test_number',
                        help='Number of test sentences, default is %(default)s')
    parser.add_argument('-D', '--dataset', type=str, default=None, dest='dataset',
                        help='Set some default datasets (dict and test file)')

    args = parser.parse_args()

    if args.dataset is not None:
        dataset = Datasets[args.dataset]

        args.dictionary_source = os.path.join('data', 'dic', dataset[6])
        args.dictionary_target = os.path.join('data', 'dic', dataset[7])
        args.source = os.path.join('data', 'test', dataset[8])

    main(args.model, args.dictionary_source, args.dictionary_target, args.source, args,
         k=args.k, normalize=args.n, chr_level=args.c)
