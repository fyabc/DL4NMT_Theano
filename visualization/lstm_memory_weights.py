#! /usr/bin/python
# -*- encoding: utf-8 -*-

"""Visualize memory weights of LSTM models."""

import argparse
import cPickle as pkl
import os
import sys
from pprint import pprint

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import theano
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from libs.config import DefaultOptions
from libs.model import build_and_init_model
from libs.utility.translate import load_translate_data
from libs.constants import Datasets

__author__ = 'fyabc'


Gates = ['input_gates_list', 'forget_gates_list', 'output_gates_list', 'state_list', 'memory_list']
GatesAtt = ['input_gates_att_list', 'forget_gates_att_list', 'output_gates_att_list']
InnerGates = Gates[:3]  # Inner gates, after sigmoid, (0.0, 1.0)
OuterGates = Gates[3:]  # Outer gates, after tanh, (-1.0, 1.0)

FontSize = 20
TextFontSize = 14
InnerColorMap = cm.bwr
OuterColorMap = cm.gray


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
        get_gates=True,
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
    for key in Gates:
        del kw_ret[key][len(real_sidx):]
        for step, weights in enumerate(kw_ret[key]):
            kw_ret[key][step] = weights[:, real_sidx[step], :]
    for key in GatesAtt:
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
            'dim': options['dim'],
        })

        tgt_seq, kw_ret = translate_sentence(src_seq, build_result, k, normalize)

        results[-1]['output'] = seq2words(tgt_seq, word_idict_trg)
        results[-1]['kw_ret'] = kw_ret
        results[-1]['n_layers'] = len(kw_ret['input_gates_list'][0])

        print 'Input:', lines[i],
        print 'Output:', results[-1]['output']
        print '=============================='

    return results


def print_results(results, args):
    for result in results:
        print '=============================='

        print 'Translating sentence {}:'.format(result['index'])
        print 'Input sentence:', result['input'],

        print 'Output sentence:', result['output']

        kw_ret = result['kw_ret']
        print 'Input gates:'
        for step, input_gates in enumerate(kw_ret['input_gates_list']):
            for layer_id, input_gate in enumerate(input_gates):
                print 'step {}, layer {}: shape = {}'.format(step, layer_id, input_gate.shape)

        print '------------------------------'
        print 'Forget gates attention:'
        for step, forget_gates_att in enumerate(kw_ret['forget_gates_att_list']):
            print 'step {}: shape = {}'.format(step, forget_gates_att.shape)


def plot_values(results, args):
    result = results[args.index]

    kw_ret = result['kw_ret']
    n_layers = result['n_layers']

    print result['output']

    words_in = result['input'].split()
    words_out = result['output'].split()
    n_in = len(words_in)
    n_out = len(words_out)
    dim = result.get('dim', len(kw_ret['input_gates_list'][0][0]))

    n_gates, n_inner_gates, n_outer_gates = len(Gates), len(InnerGates), len(OuterGates)

    # Background & Text
    plt.text(-1.0, 0.0, '$Source:$',
             fontsize=FontSize, horizontalalignment='center', verticalalignment='center')
    plt.text(-1.0, 1.0, '$Target:$',
             fontsize=FontSize, horizontalalignment='center', verticalalignment='center')
    for i in xrange(n_layers):
        plt.text(-2.0, 2.0 + n_gates // 2 + n_gates * i, '$layer_{}:$'.format(i),
                 fontsize=FontSize, horizontalalignment='center', verticalalignment='center')
        for j, gate_name in enumerate(('$i_t:$', '$f_t:$', '$o_t:$', '$h_t:$', '$c_t:$')):
            plt.text(-1.0, 1.0 + n_gates * (i + 1) - j, gate_name,
                     fontsize=FontSize, horizontalalignment='center', verticalalignment='center')

    for i, word in enumerate(words_in):
        plt.text(i, 0.0, unicode(word, encoding='utf-8'),
                 fontsize=TextFontSize, horizontalalignment='center', verticalalignment='center')
    for i, word in enumerate(words_out):
        plt.text(i, 1.0, unicode(word, encoding='utf-8'),
                 fontsize=TextFontSize, horizontalalignment='center', verticalalignment='center')

    if args.get_value == 'mean':
        value_idx = None
    elif args.get_value == 'random':
        value_idx = np.random.randint(dim)
    else:
        value_idx = int(args.get_value)

    # Prepare data
    inner_value_arrays = [np.empty((n_inner_gates, n_out + 1)) for _ in xrange(n_layers)]
    outer_value_arrays = [np.empty((n_outer_gates, n_out + 1)) for _ in xrange(n_layers)]

    def _fill_value(value_arrays, gate_names):
        for i, gate_name in enumerate(gate_names):
            print gate_name
            for step, gate_layers in enumerate(kw_ret[gate_name]):
                for layer_id, gate in enumerate(gate_layers):
                    if args.get_value == 'mean':
                        value = gate.mean()
                    else:
                        value = gate[value_idx]
                    print format(value, '.2f'), ',',
                    value_arrays[layer_id][i][step] = value
            print

    _fill_value(inner_value_arrays, InnerGates)
    _fill_value(outer_value_arrays, OuterGates)

    # Image
    for layer_id, value_array in enumerate(inner_value_arrays):
        plt.imshow(value_array, cmap=InnerColorMap, interpolation='none', vmin=0.0, vmax=1.0,
                   extent=(-0.5, n_out + 0.5,
                           n_gates * (layer_id + 1) + 1.5 - len(InnerGates), n_gates * (layer_id + 1) + 1.5))
    for layer_id, value_array in enumerate(outer_value_arrays):
        plt.imshow(value_array, cmap=OuterColorMap, interpolation='none',
                   extent=(-0.5, n_out + 0.5,
                           n_gates * layer_id + 1.5, n_gates * (layer_id + 1) + 1.5 - len(InnerGates)))

    # Set figure style
    xmin, xmax = -2, n_out
    ymin, ymax = -1, 2 + n_gates * n_layers

    plt.xticks(range(xmin, xmax))
    plt.yticks(range(ymin, ymax))

    plt.xlim(xmin=xmin - 0.5, xmax=xmax + 0.5)
    plt.ylim(ymin=ymin - 0.5, ymax=ymax + 0.5)

    plt.grid()

    plt.show()


def plot_count(results, args):
    result = results[args.index]

    kw_ret = result['kw_ret']
    n_layers = result['n_layers']

    print result['output']

    words_in = result['input'].split()
    words_out = result['output'].split()
    n_in = len(words_in)
    n_out = len(words_out)
    dim = result.get('dim', len(kw_ret['input_gates_list'][0][0]))

    n_gates, n_inner_gates = len(Gates), len(InnerGates)

    def _saturation_plot(gate_name):
        left_count = [[0 for _ in xrange(dim)] for _ in xrange(n_layers)]
        right_count = [[0 for _ in xrange(dim)] for _ in xrange(n_layers)]
        for step, gate_layers in enumerate(kw_ret[gate_name]):
            for layer_id, gate in enumerate(gate_layers):
                for i, value in enumerate(gate):
                    if value <= 0.1:
                        left_count[layer_id][i] += 1
                    elif value >= 0.9:
                        right_count[layer_id][i] += 1

        for layer_id in xrange(n_layers):
            # FIXME: add small perturbation for clearer plot of points at the same location.
            xs = [r * 1.0 / n_out + np.random.uniform(-0.01, 0.01) for r in right_count[layer_id]]
            ys = [l * 1.0 / n_out + np.random.uniform(-0.01, 0.01) for l in left_count[layer_id]]
            plt.scatter(xs, ys, label='Layer {}'.format(layer_id))

        plt.title(gate_name)
        plt.plot([0.0, 1.0], [1.0, 0.0], color='k')
        plt.xlim(xmin=0.0, xmax=1.0)
        plt.ylim(ymin=0.0, ymax=1.0)
        plt.xlabel('fraction right saturated')
        plt.ylabel('fraction left saturated')
        plt.legend()

    for i, gate_name in enumerate(InnerGates):
        plt.subplot('1{}{}'.format(n_inner_gates, i + 1))
        _saturation_plot(gate_name)

    plt.show()


def real_main(model_name, dictionary, dictionary_target, source_file, args, k=5, normalize=False, chr_level=False):
    if args.load is not None:
        with open(args.load, 'rb') as f:
            results = pkl.load(f)
    else:
        results = get_gate_weights(model_name, dictionary, dictionary_target, source_file, args,
                                   k=k, normalize=normalize, chr_level=chr_level)

    if args.dump is not None:
        with open(args.dump, 'wb') as f:
            pkl.dump(results, f)

    if args.out_mode == 'print':
        print_results(results, args)
    elif args.out_mode == 'plot_value':
        plot_values(results, args)
    elif args.out_mode == 'plot_count':
        plot_count(results, args)
    elif args.out_mode is None:
        pass
    else:
        raise ValueError('Unknown output mode: {}'.format(args.out_mode))


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
    parser.add_argument('-i', '--index', type=int, default=6, dest='index',
                        help='Index of plot sentences, default is %(default)s')
    parser.add_argument('-D', '--dataset', type=str, default=None, dest='dataset',
                        help='Set some default datasets (dict and test file)')
    parser.add_argument('-o', '--out_mode', action='store', type=str, default=None, dest='out_mode',
                        help='Output mode, default is "%(default)s", '
                             'candidates are "print", "plot_value", "plot_count"')
    parser.add_argument('-d', '--dump', metavar='FILE', action='store', type=str, default=None, dest='dump',
                        help='Dump translate result, default is %(default)s')
    parser.add_argument('-l', '--load', metavar='FILE', action='store', type=str, default=None, dest='load',
                        help='Load exist translate result, default is %(default)s')
    parser.add_argument('-V', '--value', metavar='type', action='store', type=str, default='mean', dest='get_value',
                        help='How to get value, default is %(default)s, can be set to "mean", "random" or number')

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
