#! /usr/bin/python
# -*- coding: utf-8 -*-

import argparse
from itertools import takewhile


class _Parser(argparse.ArgumentParser):
    def convert_arg_line_to_args(self, arg_line):
        return list(takewhile(lambda word: not word.startswith('#'), arg_line.split()))


def get_parser():
    parser = _Parser(
        prog='python trainer.py',
        fromfile_prefix_chars='@',
    )

    group_path = parser.add_argument_group('Data paths')
    group_path.add_argument('-d', '-dataset', action='store', default=None, dest='dataset',
                            help='Pre-defined dataset name')
    group_path.add_argument('-train-src-file', action='store', metavar='<file>', default=None, dest='train_src_file',
                            help='Training set, source file')
    group_path.add_argument('-train-trg-file', action='store', metavar='<file>', default=None, dest='train_trg_file',
                            help='Training set, target file')
    group_path.add_argument('-small-src-file', action='store', metavar='<file>', default=None, dest='small_src_file',
                            help='Small training set, source file')
    group_path.add_argument('-small-trg-file', action='store', metavar='<file>', default=None, dest='small_trg_file',
                            help='Small training set, target file')
    group_path.add_argument('-valid-src-file', action='store', metavar='<file>', default=None, dest='valid_src_file',
                            help='Validation set, source file')
    group_path.add_argument('-valid-trg-file', action='store', metavar='<file>', default=None, dest='valid_trg_file',
                            help='Validation set, target file (preprocessed)')
    group_path.add_argument('-valid-trg-orig-file', action='store', metavar='<file>', default=None, dest='valid_trg_orig_file',
                            help='Validation set, original target file')
    group_path.add_argument('-dict-src-file', action='store', metavar='<file>', default=None, dest='dict_src_file',
                            help='Dictionary, source file')
    group_path.add_argument('-dict-trg-file', action='store', metavar='<file>', default=None, dest='dict_trg_file',
                            help='Dictionary, target file')
    group_path.add_argument('-o', '-model-file', action='store', metavar='<file>', default=None, dest='model_file',
                            help='Generated model file')
    # todo: vocab map file

    group_option = parser.add_argument_group('Trainer options')
    group_option.add_argument('-r', '-restart', action='store_false', default=True, dest='reload',
                              help='Restart training, default is closed')

    group_hp = parser.add_argument_group('Hyper-parameters')
    group_hp.add_argument('-optimizer', action='store', default='adadelta',
                          help='Optimizer, default is %(default)s')

    # todo

    return parser


def parse_options(parser):
    options = parser.parse_args(['@arguments/args_sample.txt'])

    # Process some options

    # todo

    # debug
    parser.print_help()
    print options
    # end debug

    return options


__all__ = [
    'get_parser',
    'parse_options',
]
