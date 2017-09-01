#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function

import os
import re
import cPickle as pkl
import numpy as np
import subprocess
from collections import defaultdict

from .utils import prepare_data_x

__author__ = 'fyabc'


def translate(input_, model, f_init, f_next, trng, k, normalize):
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


def translate_block(input_, model, f_init, f_next, trng, k, attn_src = False):
    """Translate for batch sampler.

    :return output: a list of word indices
            all_atten_src_words: a list of attented src words for each src sentence
    """
    x, x_mask = prepare_data_x(input_, maxlen=None, pad_eos=True, pad_sos=False)

    batch_sample, batch_sample_score, sample_attn_src_words = model.gen_batch_sample(
        f_init, f_next, x, x_mask, trng,
        k=k, maxlen=200, eos_id=0, attn_src=attn_src,
    )
    assert len(batch_sample) == len(batch_sample_score)

    output = []
    all_atten_src_words = []

    for sample, sample_score, sample_attn_src_word in zip(batch_sample, batch_sample_score, sample_attn_src_words):
        score = sample_score / np.array([len(s) for s in sample])
        chosen_idx = np.argmin(score)
        output.append(sample[chosen_idx])
        if len(sample_attn_src_word) != 0:
            all_atten_src_words.append(sample_attn_src_word[chosen_idx])

    return output, all_atten_src_words

def load_zhen_trans_file(source_file, word_dict, n_words_src):
    """
    :param
    :return
    """

    source_sentences_str = []
    source_sentences_num = []
    source_hotfix = []

    for line in open(source_file, 'r'):
        if line == '':
            break
        sent = line.strip().split('||||')
        words = sent[0].strip().split()

        if len(sent) > 1:
            hotfix = defaultdict(lambda: list())
            hotfix_segs = sent[1].strip().split('}{')
            hotfix_segs[0] = hotfix_segs[0][1:]
            hotfix_segs[-1] = hotfix_segs[-1][:-1]
            for segs in hotfix_segs:
                xx = segs.strip().split(' ||| ')
                if xx[0] == xx[1]:
                    hotfix[xx[3]].append([int(xx[0]), xx[2], xx[4]])
                    words[int(xx[0])] = xx[3]
            source_hotfix.append(hotfix)
        else:
            source_hotfix.append(None)

        source_sentences_str.append(words)
        x = map(lambda w: word_dict[w] if w in word_dict else 1, words)
        x = map(lambda ii: ii if ii < n_words_src else 1, x)
        x += [0]
        source_sentences_num.append(x)

    return source_sentences_str, source_sentences_num, source_hotfix

def load_translate_data(dictionary, dictionary_target, source_file, batch_mode=False, **kwargs):
    unk_id = kwargs.pop('unk_id', 1)
    n_words_src = kwargs.pop('n_words_src', 30000)
    echo = kwargs.pop('echo', True)
    load_input = kwargs.pop('load_input', True)
    zh_en = kwargs.pop('zhen', False)

    # load source dictionary and invert
    if echo:
        print('Load and invert source dictionary...', end='')
    with open(dictionary, 'rb') as f:
        word_dict = pkl.load(f)
    word_idict = {v: k for k, v in word_dict.iteritems()}
    word_idict[0] = '<eos>'
    word_idict[unk_id] = 'UNK'
    if echo:
        print('Done')

    # load target dictionary and invert
    if echo:
        print('Load and invert target dictionary...', end='')
    with open(dictionary_target, 'rb') as f:
        word_dict_trg = pkl.load(f)
    word_idict_trg = {v: k for k, v in word_dict_trg.iteritems()}
    word_idict_trg[0] = '<eos>'
    word_idict_trg[unk_id] = 'UNK'
    if echo:
        print('Done')

    if not load_input:
        return word_dict, word_idict, word_idict_trg

    batch_size = kwargs.pop('batch_size', 30)
    all_src_str = None
    all_src_hotfixes = None

    if not zh_en:
        with open(source_file, 'r') as f:
            all_src_sent = [line.strip().split() for line in f]

        all_src_num = []

        for seg in all_src_sent:
            tmp = [word_dict.get(w, unk_id) for w in seg]
            all_src_num.append([w if w < n_words_src else unk_id for w in tmp])
    else:
        all_src_str, all_src_num, all_src_hotfixes = load_zhen_trans_file(source_file, word_dict, n_words_src)

    all_src_num_blocks = []

    m_block = (len(all_src_num) + batch_size - 1) // batch_size
    for idx in xrange(m_block):
        all_src_num_blocks.append(all_src_num[batch_size * idx: batch_size * (idx + 1)])

    return word_dict, word_idict, word_idict_trg, all_src_num_blocks, all_src_str, all_src_hotfixes, m_block


def seqs2words(caps, word_idict_trg):
    """Sequences -> Sentences

    :param caps: a list of word indices
    :param word_idict_trg: inverted target word dict
    :return: a list of sentences
    """

    capsw = []
    for cc in caps:
        ww = []
        for w in cc:
            if w == 0:
                break
            ww.append(word_idict_trg[w])
        capsw.append(' '.join(ww))
    return capsw

def idx2str_attnBasedUNKReplace(trg_idx, src_str, src_trg_table, trg_idict, attn, hotfix):
    result_trg_str = []
    trg_len = len(trg_idx)
    if hotfix is None:
        short_dic = {}
    else:
        all_vs = []
        map(lambda kv: all_vs.extend(kv[1]), hotfix.iteritems())
        short_dic = dict((ele[0], ele[1]) for ele in all_vs)

    for idx in xrange(trg_len):
        current_word = trg_idx[idx]
        if current_word == 0:
            break
        elif current_word == 1:
            selectidx = min(attn[idx], len(src_str)-1)
            if selectidx in short_dic:
                current_word = short_dic[selectidx]
            else:
                source_word = src_str[selectidx]
                current_word = src_trg_table.get(source_word, source_word)
        elif trg_idict[current_word][0] == '$' and len(trg_idict[current_word]) > 1 and hotfix is not None:
            currLabel = trg_idict[current_word]
            srcidx= min(attn[idx], len(src_str)-1)
            possible_trans = hotfix.get(currLabel, None)
            if possible_trans is None:
                current_word = src_trg_table.get(src_str[srcidx], src_str[srcidx])
            else:
                dist = [abs(vv[0] - srcidx) for vv in possible_trans]
                selectidx = np.argmin(dist)
                current_word = possible_trans[selectidx][1]
        else:
            current_word = trg_idict[current_word]
        result_trg_str.append(current_word)

    return ' '.join(result_trg_str)

def translate_whole(model, f_init, f_next, trng, dictionary, dictionary_target, source_file,
                     k=5, normalize=False, chr_level=False, src_trg_table = None, **kwargs):
    n_words_src = kwargs.pop('n_words_src', model.O['n_words_src'])
    zhen = kwargs.pop('zhen', False)
    batch_size = kwargs.pop('batch_size', 30)
    #must be in batch mode now

    word_dict, word_idict, word_idict_trg, all_src_num_blocks, all_src_str, all_src_hotfixes, m_block \
            = load_translate_data(dictionary, dictionary_target, source_file, batch_mode=True, chr_level=chr_level, n_words_src=n_words_src, batch_size = batch_size, zhen = zhen)

    print('Translating ', source_file, '...')
    all_trans = []
    all_attn_src_words = []
    for bidx, seqs in enumerate(all_src_num_blocks):
        print('\n'.join(seqs2words(seqs, word_dict)) + '\n')
        trans, src_words = translate_block(seqs, model, f_init, f_next, trng, k, attn_src = zhen)
        all_trans.extend(trans)
        if zhen:
            all_attn_src_words.extend(src_words)
        print(bidx, '/', m_block, 'Done')

    if not zhen:
        trans = seqs2words(all_trans, word_idict_trg)
    else:
        trans = [
            idx2str_attnBasedUNKReplace(trg_idx, src_str, src_trg_table, word_idict_trg, attn, hotfix)
            for (trg_idx, src_str, attn, hotfix) in zip(all_trans, all_src_str, all_attn_src_words, all_src_hotfixes)]
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
            'perl scripts/moses/multi-bleu.perl {} < {}\n'.format(ref_file, hyp_in), shell=True,
            stdout=subprocess.PIPE)
        pl_output = pl_process.stdout.read()
    elif type_in == 'string':
        pl_process = subprocess.Popen(
            'perl scripts/moses/multi-bleu.perl {}\n'.format(ref_file), shell=True, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE)
        pl_output = pl_process.communicate(hyp_in)[0]
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


def translate_dev_get_bleu(model, f_init, f_next, trng, use_noise, **kwargs):
    dataset = kwargs.pop('dataset', model.O['task'])

    # [NOTE]: Filenames here are with path prefix.
    dev1 = kwargs.pop('dev1', model.O['valid_datasets'][0])
    dev2 = kwargs.pop('dev2', model.O['valid_datasets'][2])
    dic1 = kwargs.pop('dic1', model.O['vocab_filenames'][0])
    dic2 = kwargs.pop('dic2', model.O['vocab_filenames'][1])
    zhen = kwargs.pop('zhen', False)

    use_noise.set_value(0.)

    translated_string = translate_whole(
        model, f_init, f_next, trng,
        dic1, dic2, dev1,
        k=3, batch_mode=True,
        zhen= zhen,
    )

    use_noise.set_value(1.)

    # first de-truecase, then de-bpe
    if 'tc' in dataset:
        translated_string = subprocess.Popen(
            'perl scripts/moses/detruecase.perl',
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(os.devnull, 'w'),
            shell=True,
        ).communicate(translated_string)[0]

    if 'bpe' in dataset:
        translated_string = de_bpe(translated_string)

    return get_bleu(dev2, translated_string, type_in='string')


__all__ = [
    'get_bleu',
    'de_bpe',
    'translate_dev_get_bleu',
]
