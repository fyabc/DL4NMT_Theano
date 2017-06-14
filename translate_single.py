#! /usr/bin/python
# -*- encoding: utf-8 -*-

import argparse
import cPickle as pkl
from pprint import pprint

import numpy as np
import theano

from config import DefaultOptions
from utils import load_params
from utils_fine_tune import load_translate_data, seqs2words
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

    word_dict, word_idict, word_idict_trg, input_ = load_translate_data(
        dictionary, dictionary_target, source_file,
        batch_mode=False, chr_level=chr_level, options=options,
    )

    print 'Translating ', source_file, '...'
    trans = seqs2words(
        translate_model_single(input_, model, options, k, normalize),
        word_idict_trg,
    )
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
