#! /usr/bin/python
# -*- encoding: utf-8 -*-

import argparse
import cPickle as pkl
from pprint import pprint

import numpy as np
import theano

from libs.constants import Datasets
from libs.models import build_and_init_model
from libs.utility.utils import load_options_test
from libs.utility.translate import translate_whole, chosen_by_len_alpha, get_bleu, seqs2words

def main(model, dictionary, dictionary_target, source_file, saveto, k=5,alpha = 0,
         normalize=False, chr_level=False, batch_size=1, zhen = False, src_trg_table_path = None, search_all_alphas = False, ref_file = None, dump_all = False, args = None):
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

    model, _ = build_and_init_model(model, options=options, build=False, model_type=model_type)

    f_init, f_next = model.build_sampler(trng=trng, use_noise = use_noise, batch_mode = batch_mode, dropout=options['use_dropout'], need_srcattn = zhen)

    trans, all_cand_ids, all_cand_trans, all_scores, word_idic_tgt = translate_whole(model, f_init, f_next, trng, dictionary, dictionary_target, source_file, k, normalize, alpha= alpha,
                                src_trg_table = src_trg_table, zhen = zhen, n_words_src = options['n_words_src'], echo = True, batch_size = batch_size)

    if search_all_alphas:
        all_alpha_values = 0.1 * np.array(xrange(10))
        for alpha_v in all_alpha_values:
            trans_ids = []
            for samples, sample_scores in zip(all_cand_ids, all_scores):
                trans_ids.append(samples[chosen_by_len_alpha(samples, sample_scores, alpha_v)])
            trans_strs = seqs2words(trans_ids, word_idic_tgt)
            print 'alpha %.2f, bleu %.2f'% (alpha_v, get_bleu(ref_file, '\n'.join(trans_strs), type_in = 'string'))
    else:
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
    parser.add_argument('-alpha', type=float, default=0,
                        help='The length penalty alpha, chose by p(y|x)/(5+|y|)^alpha, default to 0')
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
    parser.add_argument('-all_alphas', action="store_true", default=False,
                        help='Testing all length penalty alpha values, default to False, set to True')
    parser.add_argument('--trg_att', action='store_true', dest='trg_attention', default=False,
                        help='Use target attention, default is False, set to True')
    parser.add_argument('--ref_file', action='store', metavar='filename', dest='ref_file', type= str, help = 'The test ref file', default = None)

    parser.add_argument('model', type=str, help='The model path')
    parser.add_argument('dictionary_source', type=str, help='The source dict path')
    parser.add_argument('dictionary_target', type=str, help='The target dict path')
    parser.add_argument('source', type=str, help='The source input path')
    parser.add_argument('saveto', type=str, help='The translated file output path')
    parser.add_argument('st_table_path', nargs='?', type=str, help = 'The src tgt map file path for zhen', default= None)
    args = parser.parse_args()

    assert not args.all_alphas or args.ref_file

    main(args.model, args.dictionary_source, args.dictionary_target, args.source,
         args.saveto, k=args.k, alpha= args.alpha,normalize=args.n,
         chr_level=args.c, batch_size=args.b, args=args, src_trg_table_path= args.st_table_path if args.zhen else None, zhen= args.zhen,
         ref_file= args.ref_file, search_all_alphas = args.all_alphas,dump_all = args.all)
