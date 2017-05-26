#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

__author__ = 'fyabc'

profile = False
fX = 'float32'

ImmediateFilenameBackup = '{}_imm.pkl'
ImmediateFilename = '{}_imm.pkl.gz'
TempImmediateFilename = '{}_imm_tmp.pkl.gz'

# Cycle of shuffle data.
ShuffleCycle = 8

# Set datasets
# train1, train2, small1, small2, valid1, valid2, dic1, dic2
Datasets = {
    'en-fr': [
        'filtered_en-fr.en', 'filtered_en-fr.fr',
        'small_en-fr.en', 'small_en-fr.fr',
        'dev_en.tok', 'dev_fr.tok',
        'filtered_dic_en-fr.en.pkl', 'filtered_dic_en-fr.fr.pkl',
    ],
    'en-fr_tc': [
        'tc_filtered_en-fr.en', 'tc_filtered_en-fr.fr',
        'small_tc_en-fr.en', 'small_tc_en-fr.fr',
        'tc_dev_en.tok', 'tc_dev_fr.tok',
        'tc_filtered_en-fr.en.pkl', 'tc_filtered_en-fr.fr.pkl',
    ],

    'en-fr_bpe': [
        'en-fr_en.tok.bpe.32000', 'en-fr_fr.tok.bpe.32000',
        'small_en-fr_en.tok.bpe.32000', 'small_en-fr_fr.tok.bpe.32000',
        'dev_en-fr_en.tok.bpe.32000', 'dev_en-fr_fr.tok.bpe.32000 ',
        'en-fr_vocab.bpe.32000.pkl', 'en-fr_vocab.bpe.32000.pkl',
    ],
    'en-fr_bpe_tc': [
        'tc_en-fr_en.tok.bpe.32000', 'tc_en-fr_fr.tok.bpe.32000',
        'tc_small_en-fr_en.tok.bpe.32000', 'tc_small_en-fr_fr.tok.bpe.32000',
        'tc_dev_en-fr_en.tok.bpe.32000', 'tc_dev_en-fr_fr.tok.bpe.32000 ',
        'tc_en-fr_vocab.bpe.32000.pkl', 'tc_en-fr_vocab.bpe.32000.pkl',
    ],

    'en-de': [
        'en-de.en_0', 'en-de.de_0',
        'small_en-de.en_0', 'small_en-de.de_0',
        'dev_en.tok', 'dev_de.tok',
        'en-de.en.pkl', 'en-de.de.pkl',
    ],
    'en-de_tc': [
        '', '',
        '', '',
        '', '',
        '', '',
    ],

    'en-de_bpe': [
        '', '',
        '', '',
        '', '',
        '', '',
    ],
    'en-de_bpe_tc': [
        '', '',
        '', '',
        '', '',
        '', '',
    ],

    'en-de-s2s_bpe': [
        'train.tok.clean.bpe.32000.en', 'train.tok.clean.bpe.32000.de',
        'small_train.tok.clean.bpe.32000.en', 'small_train.tok.clean.bpe.32000.de',
        'newstest2013.tok.bpe.32000.en', 'newstest2013.tok.bpe.32000.de',
        'vocab.bpe.32000.pkl', 'vocab.bpe.32000.pkl',
    ],
    'en-de-s2s_bpe_tc': [
        '', '',
        '', '',
        '', '',
        '', '',
    ],

    'zh-en': [
        'zh-en.1.25M.zh', 'zh-en.1.25M.en',
        'small_zh-en.1.25M.zh', 'small_zh-en.1.25M.en',
        'Nist2003.chs.word.max50.snt', 'Nist2003.enu.word.max50.snt',
        'zh-en.1.25M.zh.pkl', 'zh-en.1.25M.en.pkl',
    ],
    'zh-en_tc': [
        'tc_zh-en.1.25M.zh', 'tc_zh-en.1.25M.en',
        'small_tc_zh-en.1.25M.zh', 'small_tc_zh-en.1.25M.en',
        'tc_Nist2003.chs.word.max50.snt', 'tc_Nist2003.chs.word.max50.snt',
        'tc_zh-en.1.25M.zh.pkl', 'tc_zh-en.1.25M.en.pkl',
    ],
}
