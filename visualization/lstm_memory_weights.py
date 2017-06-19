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
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config import DefaultOptions
from model import build_and_init_model
from utils_fine_tune import load_translate_data
from constants import Datasets

__author__ = 'fyabc'


def seq2words(tgt_seq, tgt_dict):
    words = []

    for w in tgt_seq:
        if w == 0:
            break
        words.append(tgt_dict[w])

    return ' '.join(words)


def translate_sentence(src_seq, build_result, k, normalize):
    model, f_init, f_next, trng = build_result

    # sample given an input sequence and obtain scores
    sample, score, kw_ret = model.gen_sample(
        f_init, f_next,
        np.array(src_seq).reshape([len(src_seq), 1]),
        trng=trng, k=k, maxlen=200,
        stochastic=False, argmax=False,
        ret_memory=True, get_gates=True,
    )

    lengths = [len(s) for s in sample]

    # normalize scores according to sequence lengths
    if normalize:
        lengths = np.array(lengths)
        score = score / lengths
    sidx = np.argmin(score)

    # FIXME: Code here suppose that the order of lived samples will not change during sampling.

    real_sidx = [0]     # Always only one init weights
    for i in xrange(1, max(lengths)):
        real_sidx_i = -1
        for j in xrange(sidx + 1):
            if lengths[j] > i:
                real_sidx_i += 1
        if real_sidx_i >= 0:
            real_sidx.append(real_sidx_i)
        else:
            break

    # Get gates of best sample
    for key in 'input_gates_list', 'forget_gates_list', 'output_gates_list':
        del kw_ret[key][len(real_sidx):]
        for step, weights in enumerate(kw_ret[key]):
            kw_ret[key][step] = weights[:, real_sidx[step], :]
    for key in 'input_gates_att_list', 'forget_gates_att_list', 'output_gates_att_list':
        # FIXME: need test
        del kw_ret[key][len(real_sidx):]
        for step, weight in enumerate(kw_ret[key]):
            kw_ret[key][step] = weight[real_sidx[step], :]

    return sample[sidx], kw_ret


def get_gate_weights(model_name, dictionary, dictionary_target, source_file, args,
                     k=5, normalize=False, chr_level=False):
    # load model model_options
    with open('%s.pkl' % model_name, 'rb') as f:
        options = DefaultOptions.copy()
        options.update(pkl.load(f))

        print 'Options:'
        pprint(options)

    word_dict, word_idict, word_idict_trg = load_translate_data(
        dictionary, dictionary_target, source_file,
        batch_mode=False, chr_level=chr_level, load_input=False
    )

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
    trng = RandomStreams(1234)
    use_noise = theano.shared(np.float32(0.))

    model, _ = build_and_init_model(model_name, options, build=False)

    f_init, f_next = model.build_sampler(
        trng=trng, use_noise=use_noise, batch_mode=False, get_gates=True,
    )
    build_result = model, f_init, f_next, trng
    print 'Done'

    results = []

    for i, src_seq in enumerate(inputs):
        results.append({
            'index': i,
            'input': lines[i],
        })

        tgt_seq, kw_ret = translate_sentence(src_seq, build_result, k, normalize)

        results[-1]['output'] = seq2words(tgt_seq, word_idict_trg)
        results[-1]['kw_ret'] = kw_ret

    return results


def print_results(results, args):
    print '=============================='

    for result in results:
        print 'Translating sentence {}:'.format(result['index'])
        print 'Input sentence:', result['input'],

        print 'Output sentence:', result['output']

        kw_ret = result['kw_ret']
        print 'Visualize LSTM memory and gates:'
        for step, input_gates in enumerate(kw_ret['input_gates_list']):
            for layer_id, input_gate in enumerate(input_gates):
                print('step {}, layer {}: shape = {}'.format(step, layer_id, input_gate.shape))

        print


def plot_results(results, args):
    pass


def real_main(model_name, dictionary, dictionary_target, source_file, args, k=5, normalize=False, chr_level=False):
    results = get_gate_weights(model_name, dictionary, dictionary_target, source_file, args,
                               k=k, normalize=normalize, chr_level=chr_level)

    if args.out_mode == 'print':
        print_results(results, args)
    elif args.out_mode == 'plot':
        plot_results(results, args)


def main():
    parser = argparse.ArgumentParser(
        description='Visualize LSTM memory weights when translating')
    parser.add_argument('model', type=str, help='The model path')
    parser.add_argument('-k', type=int, default=4,
                        help='Beam size (?), default is %(default)s, can also use 12')
    parser.add_argument('-n', action="store_true", default=False,
                        help='Use normalize, default to False, set to True')
    parser.add_argument('-c', action="store_true", default=False,
                        help='Char level model, default to False, set to True')
    parser.add_argument('--dic1', type=str, dest='dictionary_source',
                        help='The source dict path')
    parser.add_argument('--dic2', type=str, dest='dictionary_target',
                        help='The target dict path')
    parser.add_argument('--src', type=str, dest='source',
                        help='The source input path')
    parser.add_argument('-N', '--number', type=int, default=1, dest='test_number',
                        help='Number of test sentences, default is %(default)s')
    parser.add_argument('-D', '--dataset', type=str, default=None, dest='dataset',
                        help='Set some default datasets (dict and test file)')
    parser.add_argument('-o', '--out_mode', action='store', type=str, default='print', dest='out_mode',
                        help='Output mode, default is "%(default)s", candidates are "print", "plot"')

    args = parser.parse_args()

    if args.dataset is not None:
        dataset = Datasets[args.dataset]

        args.dictionary_source = os.path.join('data', 'dic', dataset[8])
        args.dictionary_target = os.path.join('data', 'dic', dataset[9])
        args.source = os.path.join('data', 'test', dataset[6])

    real_main(args.model, args.dictionary_source, args.dictionary_target, args.source, args,
              k=args.k, normalize=args.n, chr_level=args.c)


if __name__ == '__main__':
    main()
