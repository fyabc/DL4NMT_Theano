#! /usr/bin/python
# -*- coding: utf-8 -*-

import sys
import os
import argparse
from itertools import takewhile

from libs.constants import Datasets
from libs.gpu_manager import get_gpu_usage


class _Parser(argparse.ArgumentParser):
    def convert_arg_line_to_args(self, arg_line):
        return list(takewhile(lambda word: not word.startswith('#'), arg_line.split()))


def get_parser():
    parser = _Parser(
        prog='python trainer.py',
        description='Train the deep NMT model. See more information in user manual.',
        fromfile_prefix_chars='@',
    )

    group_path = parser.add_argument_group('Path options')
    group_path.add_argument('-d', '-dataset', action='store', default='en-fr', dest='dataset',
                            help='Pre-defined dataset name')
    group_path.add_argument('-train-src-file', action='store', metavar='FILE', default=None, dest='train1',
                            help='Training set, source file')
    group_path.add_argument('-train-trg-file', action='store', metavar='FILE', default=None, dest='train2',
                            help='Training set, target file')
    group_path.add_argument('-small-src-file', action='store', metavar='FILE', default=None, dest='small1',
                            help='Small training set, source file')
    group_path.add_argument('-small-trg-file', action='store', metavar='FILE', default=None, dest='small2',
                            help='Small training set, target file')
    group_path.add_argument('-valid-src-file', action='store', metavar='FILE', default=None, dest='valid1',
                            help='Validation set, source file')
    group_path.add_argument('-valid-trg-file', action='store', metavar='FILE', default=None, dest='valid2',
                            help='Validation set, target file (preprocessed)')
    group_path.add_argument('-valid-trg-orig-file', action='store', metavar='FILE', default=None,
                            dest='valid3', help='Validation set, original target file')
    group_path.add_argument('-dict-src-file', action='store', metavar='FILE', default=None, dest='dic1',
                            help='Dictionary, source file')
    group_path.add_argument('-dict-trg-file', action='store', metavar='FILE', default=None, dest='dic2',
                            help='Dictionary, target file')
    group_path.add_argument('-o', '-model-file', action='store', metavar='FILE', default=None, dest='model_file',
                            help='Generated model file')
    group_path.add_argument('-pre-load-file', action='store', metavar='FILE', default=None, dest='pre_load_file',
                            help='Pre-load model file')
    group_path.add_argument('-map-src-file', action='store', metavar='FILE', dest='src_vocab_map_file', type=str,
                            default=None, help='The file containing source vocab mapping information'
                                               'used to initialize a model on large dataset from small one')
    group_path.add_argument('-map-trg-file', action='store', metavar='FILE', dest='tgt_vocab_map_file', type=str,
                            default=None, help='The file containing target vocab mapping information'
                                               'used to initialize a model on large dataset from small one')
    group_path.add_argument('-emb', action='store', metavar='FILE', dest='given_embedding', type=str, default=None,
                            help='Given embedding model file, default is %(default)s')
    group_path.add_argument('-gpu-map-file', action='store', metavar='FILE', dest='gpu_map_file', type=str,
                            default=None, help='The file containing gpu id mapping information, '
                                               'each line is in the form physical_gpu_id\\theano_id')

    group_option = parser.add_argument_group('Trainer options')
    group_option.add_argument('-r', '-restart', action='store_false', default=True, dest='reload',
                              help='Restart training, default is closed')
    group_option.add_argument('-dump-before', action='store_true', default=False, dest='dump_before_train',
                              help='Dump model before training, default is False, set to True')
    group_option.add_argument('-S', '-not-shuffle', action='store_false', default=True, dest='shuffle',
                              help='Do not shuffle data per epoch')
    group_option.add_argument('-valid-freq', action='store', metavar='N', dest='valid_freq', type=int, default=5000,
                              help='Validation frequency, default is %(default)s')
    group_option.add_argument('-save-freq', action='store', metavar='N', default=10000, type=int, dest='save_freq',
                              help='Model save frequency, default is %(default)s')
    group_option.add_argument('-dev-bleu-freq', action='store', metavar='N', default=20000, type=int,
                              dest='dev_bleu_freq', help='Get dev set BLEU frequency, default is %(default)s')
    group_option.add_argument('-distribute', action='store', metavar='type', dest='dist_type', type=str, default=None,
                              help='The distribution version, default is None (singe GPU mode), '
                                   'candidates are "mv", "mpi_reduce"')
    group_option.add_argument('-nccl', action="store_true", default=False, dest='nccl',
                              help='Use NCCL in distributed mode, default is False, set to True')
    group_option.add_argument('-clip-grads-local', action="store_true", default=False, dest='clip_grads_local',
                              help='Clip grads in distributed mode, default is False, set to True')
    group_option.add_argument('-recover-lr-iter', action='store', metavar='lr', dest='dist_recover_lr', type=int,
                              default=10000, help='The mini-batch index to recover learning rate in distributed mode, '
                                                  'default is %(default)s')
    group_option.add_argument('-abandon-imm', action="store_true", default=False, dest='abandon_imm',
                              help='Whether to load previous immediate params, default to False, set to True')
    group_option.add_argument('-fix-rnn-weights', action="store_true", default=False, dest='fix_rnn_weights',
                              help='Fix RNN weights during training, default to False, set to True')
    group_option.add_argument('-ft-patience', action='store', metavar='N', dest='fine_tune_patience', type=int,
                              default=-1, help='Fine tune patience, default is %(default)s, set 8 to enable it')

    group_hp = parser.add_argument_group('Model hyper-parameters')
    group_hp.add_argument('-n-words-src', action='store', metavar='N', default=30000, type=int, dest='n_words_src',
                          help='Vocabularies in source side, default is %(default)s')
    group_hp.add_argument('-n-words-trg', action='store', metavar='N', default=30000, type=int, dest='n_words_tgt',
                          help='Vocabularies in target side, default is %(default)s')
    group_hp.add_argument('-optimizer', action='store', default='adadelta',
                          help='Optimizer, default is %(default)s')
    group_hp.add_argument('-lr', action="store", metavar="learning_rate", dest="learning_rate", type=float, default=1.0,
                          help='Start learning rate, default is %(default)s')
    group_hp.add_argument('-dim', action='store', metavar='N', default=512, type=int, dest='dim',
                          help='Dim of hidden units, default is %(default)s')
    group_hp.add_argument('-bs', action='store', metavar='N', default=128, type=int, dest='batch_size',
                          help='Train batch size, default is %(default)s')
    group_hp.add_argument('-valid-bs', action='store', metavar='N', default=128, type=int, dest='valid_batch_size',
                          help='Valid batch size, default is %(default)s')
    group_hp.add_argument('-dim-word', action='store', metavar='N', default=512, type=int, dest='dim_word',
                          help='Dim of word embedding, default is %(default)s')
    group_hp.add_argument('-maxlen', action='store', metavar='N', default=80, type=int, dest='maxlen',
                          help='Max sentence length, default is %(default)s')
    group_hp.add_argument('-enc', action='store', metavar='N', default=1, type=int, dest='n_encoder_layers',
                          help='Number of encoder layers, default is %(default)s')
    group_hp.add_argument('-dec', action='store', metavar='N', default=1, type=int, dest='n_decoder_layers',
                          help='Number of decoder layers, default is %(default)s')
    group_hp.add_argument('-max-epochs', action='store', metavar='N', default=100, type=int, dest='max_epochs',
                          help='Maximum epochs, default is %(default)s')
    group_hp.add_argument('-unit', action='store', metavar='unit', dest='unit', type=str, default='lstm',
                          help='The recurrent unit type, default is "%(default)s", candidates are '
                               '"lstm", "gru", "multi_lstm", "multi_gru"')
    group_hp.add_argument('-attention', action='store', metavar='N', dest='attention_layer_id', type=int, default=0,
                          help='Attention layer index, default is %(default)s')
    group_hp.add_argument('-residual-enc', action='store', metavar='type', dest='residual_enc', type=str, default=None,
                          help='Residual connection of encoder, default is %(default)s, candidates are '
                               'None, "layer_wise", "last"')
    group_hp.add_argument('-residual-dec', action='store', metavar='type', dest='residual_dec', type=str,
                          default='layer_wise', help='Residual connection of decoder, default is "layer_wise", '
                                                     'candidates are None, "layer_wise", "last"')
    group_hp.add_argument('-Z', '-not-zigzag', action='store_false', default=True, dest='use_zigzag',
                          help='Do not use zigzag trick in encoder')
    group_hp.add_argument('-dropout', action='store', metavar='dropout', dest='dropout', type=float, default=False,
                          help='Dropout rate, default is False (not use dropout)')
    group_hp.add_argument('-enc-unit-size', action='store', metavar='N', default=2, type=int, dest='unit_size',
                          help='Number of encoder unit size, default is %(default)s')
    group_hp.add_argument('-dec-unit-size', action='store', metavar='N', default=2, type=int, dest='cond_unit_size',
                          help='Number of decoder unit size, default is %(default)s')
    group_hp.add_argument('-clip', action='store', metavar='clip', dest='clip', type=float, default=1.0,
                          help='Gradient clip rate, default is %(default)s')
    group_hp.add_argument('-lr-discount', action='store', metavar='freq', dest='lr_discount_freq', type=int,
                          default=-1, help='The learning rate discount frequency, default is %(default)s')
    group_hp.add_argument('-all-att', action='store_true', dest='all_att', default=False,
                          help='Generate attention from all decoder layers, default is False, set to True')
    group_hp.add_argument('-avg-ctx', action='store_true', dest='avg_ctx', default=False,
                          help='Average all context vectors to get softmax, default is False, set to True')
    group_hp.add_argument('-trg-att', action='store', metavar='N', dest='trg_attention_layer_id', type=int, default=None,
                          help='Target attention layer id, default is None (not use target attention)')
    group_hp.add_argument('-start-epoch', action='store', default=0, type=int, dest='start_epoch',
                          help='The starting epoch, default is %(default)s')
    group_hp.add_argument('-m', '-manual', action='store_false', dest='auto', default=True,
                          help='Set some hyper-parameters manually')

    group_other = parser.add_argument_group('Other options')
    group_other.add_argument('-fix-dp-bug', action="store_true", default=False, dest='fix_dp_bug',
                             help='Fix previous dropout bug, default to False, set to True')
    group_other.add_argument('-reader-buffer-size', action='store', default=40, type=int, dest='buffer_size',
                             help='The buffer size in data reader, default is %(default)s')

    return parser


def parse_options(parser):
    args = parser.parse_args()

    # Process some options

    print args

    if args.residual_enc == 'None':
        args.residual_enc = None
    if args.residual_dec == 'None':
        args.residual_dec = None
    if args.dist_type != 'mv' and args.dist_type != 'mpi_reduce':
        args.dist_type = None

    # FIXME: Auto mode
    if args.auto:
        if args.n_encoder_layers <= 2:
            args.dropout = False
            args.clip = 1.0
        else:
            args.dropout = 0.1
            args.clip = 5.0

        if args.n_encoder_layers <= 1:
            args.residual_enc = None
        if args.n_decoder_layers <= 1:
            args.residual_dec = None
            args.attention_layer_id = 0

        args.cond_unit_size = args.unit_size

    # If dataset is not 'en-fr', old value of dataset options like 'args.train1' will be omitted
    if args.dataset != 'en-fr':
        args.train1, args.train2, args.small1, args.small2, args.valid1, args.valid2, args.valid3, test1, test2, \
            args.dic1, args.dic2 = Datasets[args.dataset]

    # Connection type fix to 2 (bidirectional only at first layer)
    args.connection_type = 2

    print 'Command line arguments:'
    print args
    sys.stdout.flush()

    # Init multiverso or mpi and set theano flags.
    if args.dist_type == 'mv':
        try:
            import multiverso as mv
        except ImportError:
            import libs.multiverso_ as mv

        # FIXME: This must before the import of theano!
        mv.init(sync=True)
        worker_id = mv.worker_id()
        workers_cnt = mv.workers_num()
    elif args.dist_type == 'mpi_reduce':
        from mpi4py import MPI

        communicator = MPI.COMM_WORLD
        worker_id = communicator.Get_rank()
        workers_cnt = communicator.Get_size()

    if args.dist_type:
        available_gpus = get_gpu_usage(workers_cnt)
        gpu_maps_info = {idx: idx for idx in available_gpus}
        if args.gpu_map_file:
            for line in open(os.path.join('resources', args.gpu_map_file), 'r'):
                phy_id, theano_id = line.split()
                gpu_maps_info[int(phy_id)] = int(theano_id)
        theano_id = gpu_maps_info[available_gpus[worker_id]]
        print 'worker id:%d, using theano id:%d, physical id %d' % (worker_id, theano_id, available_gpus[worker_id])
        os.environ['THEANO_FLAGS'] = 'device=cuda{},floatX=float32'.format(theano_id)
        sys.stdout.flush()

    return args


def run(args):
    from libs.nmt import train

    train(
        max_epochs=args.max_epochs,
        saveto=args.model_file,
        preload=args.pre_load_file,
        reload_=args.reload,
        dim_word=args.dim_word,
        dim=args.dim,
        decay_c=0.,
        clip_c=args.clip,
        lrate=args.learning_rate,
        optimizer=args.optimizer,
        maxlen=args.maxlen,
        batch_size=args.batch_size,
        valid_batch_size=args.valid_batch_size,
        dispFreq=1,
        saveFreq=args.save_freq,
        validFreq=args.valid_freq,
        datasets=('./data/train/{}'.format(args.train1),
                  './data/train/{}'.format(args.train2)),
        valid_datasets=('./data/dev/{}'.format(args.valid1),
                        './data/dev/{}'.format(args.valid2),
                        './data/dev/{}'.format(args.valid3)),
        small_train_datasets=('./data/train/{}'.format(args.small1),
                              './data/train/{}'.format(args.small2)),
        vocab_filenames=('./data/dic/{}'.format(args.dic1),
                         './data/dic/{}'.format(args.dic2)),
        task=args.dataset,
        use_dropout=args.dropout,
        overwrite=False,
        n_words=args.n_words_tgt,
        n_words_src=args.n_words_src,

        dump_before_train=args.dump_before_train,
        plot_graph=None,
        lr_discount_freq=args.lr_discount_freq,

        n_encoder_layers=args.n_encoder_layers,
        n_decoder_layers=args.n_decoder_layers,
        encoder_many_bidirectional=args.connection_type == 1,

        attention_layer_id=args.attention_layer_id,
        unit=args.unit,
        residual_enc=args.residual_enc,
        residual_dec=args.residual_dec,
        use_zigzag=args.use_zigzag,
        given_embedding=args.given_embedding,

        unit_size=args.unit_size,
        cond_unit_size=args.cond_unit_size,

        given_imm=not args.abandon_imm,
        dump_imm=True,
        shuffle_data=args.shuffle,

        decoder_all_attention=args.all_att,
        average_context=args.avg_ctx,

        dist_type=args.dist_type,
        dist_recover_lr_iter=args.dist_recover_lr,

        fine_tune_patience=args.fine_tune_patience,
        nccl=args.nccl,
        src_vocab_map_file=args.src_vocab_map_file,
        tgt_vocab_map_file=args.tgt_vocab_map_file,

        trg_attention_layer_id=args.trg_attention_layer_id,
        dev_bleu_freq=args.dev_bleu_freq,
        fix_dp_bug=args.fix_dp_bug,
        io_buffer_size=args.buffer_size,
        start_epoch=args.start_epoch,
        fix_rnn_weights=args.fix_rnn_weights,
    )


__all__ = [
    'get_parser',
    'parse_options',
    'run',
]
