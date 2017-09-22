#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Test the deliberation model."""

import argparse

import numpy as np

from libs.run.run_PR import predict


def main():
    parser = argparse.ArgumentParser(description='Test the deliberation model.')
    parser.add_argument('modelpath', help='The model path, default is %(default)s')
    parser.add_argument('--log', action='store', dest='logfile', default=None, metavar='FILE',
                        help='The logging filename, default is %(default)s')
    parser.add_argument('--action', action='store', dest='action', default='aprs',
                        help='Test action (string contains "a", "p", "r" and "s"), default is %(default)s')
    parser.add_argument('--bs', action='store', dest='valid_batch_size', default=80, type=int, metavar='N',
                        help='Batch size, default is %(default)s')
    parser.add_argument('--start', action='store', dest='start_idx', default=10000, type=int, metavar='N',
                        help='Start iteration of model, default is %(default)s')
    parser.add_argument('--step', action='store', dest='step_idx', default=10000, type=int, metavar='N',
                        help='Step iteration of model, default is %(default)s')
    parser.add_argument('--end', action='store', dest='end_idx', default=10000, type=int, metavar='N',
                        help='End iteration of model, default is %(default)s')
    parser.add_argument('-k', action='store', dest='k_list', default=['1'], metavar='N', nargs='+',
                        help='Get the top-k of word probabilities, k is a list, default is %(default)s')

    args = parser.parse_args()

    new_k_list = []
    for k_str in args.k_list:
        if '..' in k_str:
            start, end = k_str.split('..')
            new_k_list.extend(range(int(start), int(end) + 1))
        else:
            new_k_list.append(int(k_str))

    args.k_list = np.unique(new_k_list)

    print args

    # todo: need test on pre-trained models
    predict(
        modelpath=args.modelpath,
        action=args.action,
        valid_batch_size=args.valid_batch_size,
        valid_datasets=None,
        dictionary=None,
        dictionary_target=None,
        start_idx=args.start_idx,
        step_idx=args.step_idx,
        end_idx=args.end_idx,
        logfile=args.logfile,
        k_list=args.k_list,
    )


if __name__ == '__main__':
    main()
