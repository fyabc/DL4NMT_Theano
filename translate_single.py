#! /usr/bin/python
# -*- encoding: utf-8 -*-

import argparse
import os
import re
import cPickle as pkl
from pprint import pprint

import numpy as np
import theano

from libs.config import DefaultOptions
from libs.models import build_and_init_model
from libs.utility.utils import load_options_test
from libs.utility.translate import load_translate_data, seqs2words, translate, translate_block, translate_whole

def translate_model_single(input_, model_name, options, k, normalize):
    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
    trng = RandomStreams(1234)
    use_noise = theano.shared(np.float32(0.))

    model, _ = build_and_init_model(model_name, options=options, build=False)

    # word index
    f_init, f_next = model.build_sampler(trng=trng, use_noise=use_noise)

    return translate(input_, model, f_init, f_next, trng, k, normalize)

def main(model, dictionary, dictionary_target, source_file, saveto, k=5,
         normalize=False, chr_level=False, batch_size=1, zhen = False, src_trg_table_path=None, dump_all = False, args=None):
    batch_mode = batch_size > 1
    assert batch_mode

    # load model model_options
    options = load_options_test(model)

    src_trg_table = None
    if src_trg_table_path:
        with open(src_trg_table_path, 'rb') as f:
            src_trg_table = pkl.load(f)

    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
    trng = RandomStreams(1234)
    use_noise = theano.shared(np.float32(0.))

    model_type = 'NMTModel'
    if args.trg_attention:
        model_type = 'TrgAttnNMTModel'
    if options['use_delib']:
        model_type = 'DelibNMT'

    model, _ = build_and_init_model(model, options=options, build=False, model_type=model_type)

    f_init, f_next = model.build_sampler(trng=trng, use_noise = use_noise, batch_mode = batch_mode, dropout=options['use_dropout'])

    trans, all_cand_trans = translate_whole(model, f_init, f_next, trng, dictionary, dictionary_target, source_file, k, normalize,
                                src_trg_table = src_trg_table, zhen= zhen, n_words_src = options['n_words_src'], echo= True, batch_size= batch_size)
    with open(saveto, 'w') as f:
        print >> f, '\n'.join(trans)
    if dump_all:
        saveto_dump_all = '%s.all_beam%d' % (saveto, k)
        with open(saveto_dump_all, 'w') as f:
            print >> f, '\n'.join(all_cand_trans)
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
    parser.add_argument('-zhen', action="store_true", default=False,
                        help='Whether zhen translation, default to False, set to True')
    parser.add_argument('-b', type=int, default=-1,
                        help='Batch size, default to -1, means not to use batch mode')
    parser.add_argument('-all', action="store_true", default=False,
                        help='Dump all candidate translations, default to False, set to True')
    parser.add_argument('--trg_att', action='store_true', dest='trg_attention', default=False,
                        help='Use target attention, default is False, set to True')

    parser.add_argument('model', type=str, help='The model path')
    parser.add_argument('dictionary_source', type=str, help='The source dict path')
    parser.add_argument('dictionary_target', type=str, help='The target dict path')
    parser.add_argument('source', type=str, help='The source input path')
    parser.add_argument('saveto', type=str, help='The translated file output path')
    parser.add_argument('st_table_path', nargs='?', type=str, help = 'The src tgt map file path for zhen', default= None)

    args = parser.parse_args()

    main(args.model, args.dictionary_source, args.dictionary_target, args.source,
         args.saveto, k=args.k, normalize=args.n,
         chr_level=args.c, batch_size=args.b, args=args, src_trg_table_path= args.st_table_path if args.zhen else None, zhen= args.zhen, dump_all= args.all)
