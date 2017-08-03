#! /usr/bin/python
# -*- coding: utf-8 -*-

import argparse


class _Parser(argparse.ArgumentParser):
    def convert_arg_line_to_args(self, arg_line):
        return arg_line.split()


def get_parser():
    parser = _Parser(
        prog='python trainer.py',
        fromfile_prefix_chars='@',
    )

    group_path = parser.add_argument_group('File paths', 'Specify file paths')
    group_path.add_argument('-train-src-file', action='store', metavar='<file>', default=None, dest='train_src_file',
                            help='Training set, source file')
    group_path.add_argument('-train-trg-file', action='store', metavar='<file>', default=None, dest='train_trg_file',
                            help='Training set, target file')
    group_path.add_argument('-valid-src-file', action='store', metavar='<file>', default=None, dest='valid_src_file',
                            help='Validation set, source file')
    group_path.add_argument('-valid-trg-file', action='store', metavar='<file>', default=None, dest='valid_trg_file',
                            help='Validation set, target file')

    # todo

    return parser


def parse_options(parser):
    options = parser.parse_args()

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
