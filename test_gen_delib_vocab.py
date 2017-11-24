#! /usr/bin/python
# -*- coding: utf-8 -*-

import argparse

from libs.run.gen_delib_vocab import generate


def main():
    parser = argparse.ArgumentParser(description='Generate the deliberation vocabulary.')
    parser.add_argument('model_path', help='The deliberation model path')
    parser.add_argument('dump_path', help='The path to dump the model')
    parser.add_argument('-k', action='store', dest='k', type=int, default=100,
                        help='Store top-k vocabulary, default is %(default)s')
    parser.add_argument('--bs', action='store', dest='test_batch_size', default=80, type=int, metavar='N',
                        help='Batch size, default is %(default)s')

    args = parser.parse_args()
    print args

    generate(args.model_path, args.dump_path, args.k, args.test_batch_size)


if __name__ == '__main__':
    main()
