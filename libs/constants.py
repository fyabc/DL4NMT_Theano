#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

profile = False
fX = 'float32'

ImmediateFilename = '{}_imm.iter{}.npz'
TempImmediateFilename = '{}_imm_tmp.iter{}.npz'
BestImmediateFilename = '{}_imm.npz'
BestTempImmediateFilename = '{}_imm_tmp.npz'

# Cycle of shuffle data.
ShuffleCycle = 7

# Load the N-th previous saved model when NaN detected.
NaNReloadPrevious = 3

# Set datasets
#For datasets other than zhen: train1, train2, small1, small2, valid1, valid2(postprocessed, e.g., bpe and truecase), valid3(original), test1, test2, dic1, dic2
#For zhen dataset: train1, train2, small1, small2, valid1, src_tgt_table, valid2, test1, test2, dict1,dict2
Datasets = {
    'en-fr': [
        'filtered_en-fr.en', 'filtered_en-fr.fr',
        'small_en-fr.en', 'small_en-fr.fr',
        'dev_en.tok', 'dev_fr.tok', 'dev_fr.tok',
        'test_en-fr.en.tok','test_en-fr.fr.tok',
        'filtered_dic_en-fr.en.pkl', 'filtered_dic_en-fr.fr.pkl',
    ],
    'en-fr_tc': [
        'tc_filtered_en-fr.en', 'tc_filtered_en-fr.fr',
        'small_tc_en-fr.en', 'small_tc_en-fr.fr',
        'tc_dev_en.tok', 'tc_dev_fr.tok', 'dev_fr.tok',
        '', '',
        'tc_filtered_en-fr.en.pkl', 'tc_filtered_en-fr.fr.pkl',
    ],

    'en-fr_bpe': [
        'en-fr_en.tok.bpe.32000', 'en-fr_fr.tok.bpe.32000',
        'small_en-fr_en.tok.bpe.32000', 'small_en-fr_fr.tok.bpe.32000',
        'dev_en-fr_en.tok.bpe.32000', 'dev_en-fr_fr.tok.bpe.32000', 'dev_fr.tok',
        'test_en-fr.en.tok.bpe.32000', 'test_en-fr.fr.tok',
        'en-fr_vocab.bpe.32000.pkl', 'en-fr_vocab.bpe.32000.pkl',
    ],
    'en-fr_bpe_tc': [
        'tc_en-fr_en.tok.bpe.32000', 'tc_en-fr_fr.tok.bpe.32000',
        'tc_small_en-fr_en.tok.bpe.32000', 'tc_small_en-fr_fr.tok.bpe.32000',
        'tc_dev_en-fr_en.tok.bpe.32000', 'tc_dev_en-fr_fr.tok.bpe.32000','dev_fr.tok',
        'tc_test_en-fr.en.tok.bpe.32000', 'test_en-fr.fr.tok',
        'tc_en-fr_vocab.bpe.32000.pkl', 'tc_en-fr_vocab.bpe.32000.pkl',
    ],

    'large_en-fr_bpe_tc': [
        'tc_train_enfr_large_bpe.en', 'tc_train_enfr_large_bpe.fr',
        'tc_small_train_enfr_large_bpe.en', 'tc_small_train_enfr_large_bpe.fr',
        'tc_valid_enfr_bpe_by_large.en', 'tc_valid_enfr_bpe_by_large.fr', 'dev_fr.tok',
        'tc_test_enfr_bpe_by_large.en', 'test_en-fr.fr.tok',
        'tc_enfr_large_bpe.vocab.pkl', 'tc_enfr_large_bpe.vocab.pkl',
    ],

    'large_fr-en_bpe_tc': [
        'tc_train_enfr_large_bpe.fr', 'tc_train_enfr_large_bpe.en',
        'tc_small_train_enfr_large_bpe.fr', 'tc_small_train_enfr_large_bpe.en',
        'tc_valid_enfr_bpe_by_large.fr', 'tc_valid_enfr_bpe_by_large.en', 'dev_en.tok',
        'tc_test_enfr_bpe_by_large.fr', 'test_en-fr.en.tok',
        'tc_enfr_large_bpe.vocab.pkl', 'tc_enfr_large_bpe.vocab.pkl',
    ],

    'fr-en_bpe_tc': [
        'tc_en-fr_fr.tok.bpe.32000','tc_en-fr_en.tok.bpe.32000',
        'tc_small_en-fr_fr.tok.bpe.32000', 'tc_small_en-fr_en.tok.bpe.32000',
        'tc_dev_en-fr_fr.tok.bpe.32000', 'tc_dev_en-fr_en.tok.bpe.32000', 'dev_en.tok',
        'tc_test_en-fr.fr.tok.bpe.32000', 'test_en-fr.en.tok',
        'tc_en-fr_vocab.bpe.32000.pkl', 'tc_en-fr_vocab.bpe.32000.pkl',
    ],


    'en-de': [
        'en-de.en_0', 'en-de.de_0',
        'small_en-de.en_0', 'small_en-de.de_0',
        'dev_en.tok', 'dev_de.tok', '',
        '','',
        'en-de.en.pkl', 'en-de.de.pkl',
    ],
    'en-de_tc': [
        '', '',
        '', '',
        '', '', '',
        '', '',
        '', '',
    ],

    'en-de_bpe': [
        'en-de_en.tok.bpe.32000', 'en-de_de.tok.bpe.32000',
        'small_en-de_en.tok.bpe.32000', 'small_en-de_de.tok.bpe.32000',
        'dev_en-de_en.tok.bpe.32000', 'dev_en-de_de.tok.bpe.32000', '',
        'test_en-de.en.tok.bpe.32000', 'test_en-de.de.tok.bpe.32000',
        'en-de_vocab.bpe.32000.pkl', 'en-de_vocab.bpe.32000.pkl',
    ],
    'en-de_bpe_tc': [
        '', '',
        '', '',
        '', '', '',
        '', '',
        '', '',
    ],

    'en-de-s2s_bpe': [
        'train.tok.clean.bpe.32000.en', 'train.tok.clean.bpe.32000.de',
        'small_train.tok.clean.bpe.32000.en', 'small_train.tok.clean.bpe.32000.de',
        'newstest2013.tok.bpe.32000.en', 'newstest2013.tok.bpe.32000.de', 'dev_de.tok',
        'newstest2014.tok.bpe.32000.en', 'test_en-de.de.tok',
        'vocab.bpe.32000.pkl', 'vocab.bpe.32000.pkl',
    ],

    'en-de-s2s_bpe_tc': [
        'tc_train.tok.clean.bpe.32000.en', 'tc_train.tok.clean.bpe.32000.de',
        'tc_small_train.tok.clean.bpe.32000.en', 'tc_small_train.tok.clean.bpe.32000.de',
        'tc_newstest2013.tok.bpe.32000.en', 'tc_newstest2013.tok.bpe.32000.de', '',
        'tc_newstest2014.tok.bpe.32000.en', 'test_en-de.de.tok',
        'ende_s2s_vocab.bpe.32000.pkl', 'ende_s2s_vocab.bpe.32000.pkl',
    ],

    'de-en':[
       'train.de-en.de', 'train.de-en.en',
        'small_train.de-en.de','small_train.de-en.en',
        'dev.de-en.de', 'dev.de-en.en','dev.de-en.en',
        'test.de-en.de','test.de-en.en',
        'de-en_vocab.de.pkl','de-en_vocab.en.pkl',
    ],

    'en-de_small':[
       'train.de-en.en', 'train.de-en.de',
       'small_train.de-en.en', 'small_train.de-en.de',
       'dev.de-en.en', 'dev.de-en.de', 'dev.de-en.de',
       'test.de-en.en', 'test.de-en.de',
       'de-en_vocab.en.pkl', 'de-en_vocab.de.pkl',
    ],

    'de-en_bpe':[
       'train.de-en.bpe.25000.de', 'train.de-en.bpe.25000.en',
        'small_train.de-en.bpe.25000.de', 'small_train.de-en.bpe.25000.en',
        'dev.de-en.bpe.25000.de', 'dev.de-en.bpe.25000.en','dev.de-en.en',
        'test.de-en.bpe.25000.de', 'test.de-en.en',
        'de-en_vocab.bpe.25000.pkl', 'de-en_vocab.bpe.25000.pkl',
    ],

    'en-de_small_bpe' :[
       'train.de-en.bpe.25000.en', 'train.de-en.bpe.25000.de',
       'small_train.de-en.bpe.25000.en', 'small_train.de-en.bpe.25000.de',
       'dev.de-en.bpe.25000.en', 'dev.de-en.bpe.25000.de', 'dev.de-en.de',
       'test.de-en.bpe.25000.en', 'test.de-en.de',
       'de-en_vocab.bpe.25000.pkl', 'de-en_vocab.bpe.25000.pkl',
    ],

    'zh-en': [
        'zh-en.1.25M.zh', 'zh-en.1.25M.en',
        'small_zh-en.1.25M.zh', 'small_zh-en.1.25M.en',
        'Nist2003.chs.word.max50.snt', 'Nist2003.enu.word.max50.snt', 'Nist2003.enu.word.max50.snt',
        '','',
        'zh-en.1.25M.zh.pkl', 'zh-en.1.25M.en.pkl',
    ],
    'zh-en_1.25m': [
        'zh-en.1.25M.zh', 'zh-en.1.25M.en',
        'small_tc_zh-en.1.25M.zh', 'small_tc_zh-en.1.25M.en',
        'sorted_Nist2005.dev.txt.zh.tok', 'zh2en.giza.pkl', 'NIST2005.reference',
        'sorted_Nist2006.dev.txt.zh.tok, sorted_Nist2008.dev.txt.zh.tok, sorted_Nist2012.dev.txt.zh.tok', 'NIST2006.reference, NIST2008.reference, NIST2012.reference',
        'tc_zh-en.1.25M.zh.pkl', 'tc_zh-en.1.25M.en.pkl',
    ],
}

# Average (word) lengths for source and target in training data
# TODO: Add more datasets information here
AverageLength = {
    'en-fr': [20.671612554135102, 23.431791903792313],
    'en-fr_tc': [20.671612554135102, 23.431791903792313],
    'en-fr_bpe': [23.189612575725693, 26.158710469563228],
    'en-fr_bpe_tc': [23.189612575725693, 26.158710469563228],
    'large_en-fr_bpe_tc': [1.0, 1.0],
    'large_fr-en_bpe_tc': [1.0, 1.0],
    'fr-en_bpe_tc': [26.158710469563228, 23.189612575725693],
    'en-de': [22.812364291264533, 21.691388538100615],
    'en-de_tc': [22.812364291264533, 21.691388538100615],
    'en-de_bpe': [29.447903019762666, 30.374680577009659],
    'en-de_bpe_tc': [29.447903019762666, 30.374680577009659],
    'en-de-s2s_bpe': [28.523922420209349, 29.577490698663354],
    'en-de-s2s_bpe_tc': [28.523922420209349, 29.577490698663354],
    'de-en': [17.527614364165242, 18.500189139480582],
    'en-de_small': [18.500189139480582, 17.527614364165242],
    'de-en_bpe': [1.0, 1.0],
    'en-de_small_bpe': [1.0, 1.0],
    'zh-en': [20.524633600000001, 21.663608799999999],
    'zh-en_1.25m': [20.524633600000001, 21.663608799999999],
}
