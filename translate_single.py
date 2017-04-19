#! /usr/bin/python
# -*- encoding: utf-8 -*-

import argparse
import cPickle as pkl
from pprint import pprint

import numpy as np
import theano

from config import DefaultOptions
from utils import load_params
from model import NMTModel


__author__ = 'fyabc'


def translate_model_single(input_, model_name, options, k, normalize):
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

    def _translate(seq):
        # sample given an input sequence and obtain scores
        sample, score = model.gen_sample(
            f_init, f_next,
            np.array(seq).reshape([len(seq), 1]),
            trng=trng, k=k, maxlen=200,
            stochastic=False, argmax=False,
        )

        # normalize scores according to sequence lengths
        if normalize:
            lengths = np.array([len(s) for s in sample])
            score = score / lengths
        sidx = np.argmin(score)
        return sample[sidx]

    output = []

    for idx, x in enumerate(input_):
        print idx

        output.append(_translate(x))

    return output


def main(model, dictionary, dictionary_target, source_file, saveto, k=5,
         normalize=False, chr_level=False):
    # load model model_options
    with open('%s.pkl' % model, 'rb') as f:
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

    input_ = []

    print 'Loading input...',
    with open(source_file, 'r') as f:
        for idx, line in enumerate(f):
            if chr_level:
                words = list(line.decode('utf-8').strip())
            else:
                words = line.strip().split()

            x = [word_dict[w] if w in word_dict else 1 for w in words]
            x = [ii if ii < options['n_words_src'] else 1 for ii in x]
            x.append(0)

            input_.append(x)
    print 'Done'

    # utility function
    def _seqs2words(caps):
        capsw = []
        for cc in caps:
            ww = []
            for w in cc:
                if w == 0:
                    break
                ww.append(word_idict_trg[w])
            capsw.append(' '.join(ww))
        return capsw

    print 'Translating ', source_file, '...'
    trans = _seqs2words(translate_model_single(input_, model, options, k, normalize))
    with open(saveto, 'w') as f:
        print >> f, '\n'.join(trans)
    print 'Done'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Translate the source language test file to target language with given model (single thread)')
    parser.add_argument('-k', type=int, default=4,
                        help='Beam size (?), default to 4, can also use 12')
    parser.add_argument('-p', type=int, default=5,
                        help='Number of parallel processes, default to 5')
    parser.add_argument('-n', action="store_true", default=False,
                        help='Use normalize, default to False, set to True')
    parser.add_argument('-c', action="store_true", default=False,
                        help='Char level model, default to False, set to True')
    parser.add_argument('model', type=str, help='The model path')
    parser.add_argument('dictionary_source', type=str, help='The source dict path')
    parser.add_argument('dictionary_target', type=str, help='The target dict path')
    parser.add_argument('source', type=str, help='The source input path')
    parser.add_argument('saveto', type=str, help='The translated file output path')

    args = parser.parse_args()

    main(args.model, args.dictionary_source, args.dictionary_target, args.source,
         args.saveto, k=args.k, normalize=args.n,
         chr_level=args.c)
