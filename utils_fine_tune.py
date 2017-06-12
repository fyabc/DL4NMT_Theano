#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function

import os
import re
import cPickle as pkl
import numpy as np
import subprocess

from constants import Datasets

__author__ = 'fyabc'


def _translate(input_, model, f_init, f_next, trng, k, normalize):
    def _trans(seq):
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
        output.append(_trans(x))

    return output


def _translate_whole(model, f_init, f_next, trng, dictionary, dictionary_target, source_file,
                     k=5, normalize=False, chr_level=False, **kwargs):
    n_words_src = kwargs.pop('n_words_src', )

    # load source dictionary and invert
    with open(dictionary, 'rb') as f:
        word_dict = pkl.load(f)
    word_idict = dict()
    for kk, vv in word_dict.iteritems():
        word_idict[vv] = kk
    word_idict[0] = '<eos>'
    word_idict[1] = 'UNK'

    # load target dictionary and invert
    with open(dictionary_target, 'rb') as f:
        word_dict_trg = pkl.load(f)
    word_idict_trg = dict()
    for kk, vv in word_dict_trg.iteritems():
        word_idict_trg[vv] = kk
    word_idict_trg[0] = '<eos>'
    word_idict_trg[1] = 'UNK'

    input_ = []

    # Loading input
    with open(source_file, 'r') as f:
        for idx, line in enumerate(f):
            if chr_level:
                words = list(line.decode('utf-8').strip())
            else:
                words = line.strip().split()

            x = [word_dict[w] if w in word_dict else 1 for w in words]
            x = [ii if ii < n_words_src else 1 for ii in x]
            x.append(0)

            input_.append(x)

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

    # Translate file
    trans = _seqs2words(_translate(input_, model, f_init, f_next, trng, k, normalize))

    return '\n'.join(trans) + '\n'


def get_bleu(ref_file, hyp_in=None, type_in='filename'):
    """Get BLEU score, it will call script 'multi-bleu.perl'.

    :param ref_file: standard test filename of target language.
    :param hyp_in: input from _translate_whole script.
    :param type_in: input type, default is 'filename', can be 'filename' or 'string'.
    :return:
    """

    if type_in == 'filename':
        pl_process = subprocess.Popen(
            'perl multi-bleu.perl {} < {}\n'.format(ref_file, hyp_in), shell=True,
            stdout=subprocess.PIPE, stderr=open(os.devnull, 'w'))
        pl_output = pl_process.stdout.read()
    elif type_in == 'string':
        pl_process = subprocess.Popen(
            'perl multi-bleu.perl {}\n'.format(ref_file), shell=True, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=open(os.devnull, 'w'))
        pl_output = pl_process.communicate(hyp_in)
    else:
        raise ValueError('Wrong type_in')

    contents = pl_output.split(',')
    if len(contents) == 0:
        return 0.0
    var = contents[0].split(" = ")
    if len(var) <= 1:
        return 0.0
    BLEU = var[1]

    return float(BLEU)


def de_bpe(input_str):
    return re.sub(r'(@@ )|(@@ ?$)', '', input_str)


def translate_dev_get_bleu(model, f_init, f_next, trng, dataset, n_words_src):
    train1, train2, small1, small2, dev1, dev2, test1, test2, dic1, dic2 = Datasets[dataset]

    translated_string = _translate_whole(
        model, f_init, f_next, trng,
        './data/dic/{}'.format(dic1),
        './data/dic/{}'.format(dic2),
        './data/test/{}'.format(test1),
        k=2, n_words_src=n_words_src,
    )

    # first de-truecase, then de-bpe
    if 'tc' in dataset:
        translated_string = subprocess.Popen(
            'perl detruecase.perl',
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(os.devnull, 'w'),
            shell=True,
        ).communicate(translated_string)

    if 'bpe' in dataset:
        translated_string = de_bpe(translated_string)

    return get_bleu('./data/test/{}'.format(test2), translated_string, type_in='string')


__all__ = [
    'get_bleu',
    'de_bpe',
    'translate_dev_get_bleu',
]
