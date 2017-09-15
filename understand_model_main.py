#! /usr/bin/python
# -*- coding: utf-8 -*-

import argparse

from libs.understand_model.count_operations import real_main

__author__ = 'fyabc'


def main():
    parser = argparse.ArgumentParser(description='Analyze and understand NMT model.')
    parser.add_argument('modelpath', help='Model path')
    parser.add_argument('-d', '--dest', dest='dest', default='probs', choices=['probs'],
                        help='The function to be analyzed, default is %(default)s')

    args = parser.parse_args()
    print args

    real_main(args)


if __name__ == '__main__':
    main()
