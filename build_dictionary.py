#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function

import numpy

try:
    import cPickle as pkl
except:
    import pickle as pkl

import sys
import os
import fileinput

from collections import OrderedDict

__author__ = 'fyabc'


def main():
    for filename in sys.argv[1:]:
        src_filename = os.path.join('data', 'train', filename)
        tgt_filename = os.path.join('data', 'dic', filename)

        print('Processing', src_filename)

        word_freqs = OrderedDict()
        with open(src_filename, 'r') as f:
            for line in f:
                words_in = line.strip().split(' ')
                for w in words_in:
                    if w not in word_freqs:
                        word_freqs[w] = 0
                    word_freqs[w] += 1
        words = list(word_freqs.keys())
        freqs = list(word_freqs.values())

        sorted_idx = numpy.argsort(freqs)
        sorted_words = [words[ii] for ii in sorted_idx[::-1]]

        worddict = OrderedDict()
        worddict['eos'] = 0
        worddict['UNK'] = 1
        for ii, ww in enumerate(sorted_words):
            worddict[ww] = ii+2

        with open('%s.pkl' % tgt_filename, 'wb') as f:
            pkl.dump(worddict, f)

        print('Dump to', tgt_filename)

if __name__ == '__main__':
    main()
