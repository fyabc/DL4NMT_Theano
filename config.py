#! /usr/bin/python
# -*- coding: utf-8 -*-

__author__ = 'fyabc'

# Dict of default options (Copied from nmt.py)
DefaultOptions = dict(
    dim_word=100,  # word vector dimensionality
    dim=1000,  # the number of LSTM units
    encoder='gru',
    decoder='gru_cond',
    n_words_src=30000,
    n_words=30000,
    patience=10,  # early stopping patience
    max_epochs=5000,
    finish_after=10000000,  # finish after this many updates
    dispFreq=100,
    decay_c=0.,  # L2 regularization penalty
    alpha_c=0.,  # alignment regularization
    clip_c=-1.,  # gradient clipping threshold
    lrate=1.,  # learning rate
    maxlen=100,  # maximum length of the description
    optimizer='rmsprop',
    batch_size=16,
    saveto='model.npz',
    saveFreq=1000,  # save the parameters after every saveFreq updates
    datasets=('/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.en.tok',
              '/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.fr.tok'),
    picked_train_idxes_file=r'',
    use_dropout=False,
    reload_=False,
    overwrite=False,
    preload='',
    sort_by_len=False,

    # Options below are from v-yanfa
    convert_embedding=True,
    dump_before_train=False,
    plot_graph=None,
    vocab_filenames=('./data/dic/filtered_dic_en-fr.en.pkl',
                     './data/dic/filtered_dic_en-fr.fr.pkl'),
    map_filename='./data/dic/mapFullVocab2Top1MVocab.pkl',
    lr_discount_freq=80000,

    # Options of deeper encoder and decoder
    n_encoder_layers=1,
    n_decoder_layers=1,

    # The connection type:
    #     1. encoder_many_bidirectional = True (default):
    #         forward1 -> forward2 -> forward3 -> ...
    #         backward1 -> backward2 -> backward3 -> ...
    #     2. encoder_many_bidirectional = False:
    #         forward1 + backward1 -> forward2 -> forward3 -> ...
    encoder_many_bidirectional=True,
)
