#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Test the deliberation model."""

import argparse

from libs.run.run_deliberation import predict
from libs.constants import Datasets


def main():
    parser = argparse.ArgumentParser(description='Test the deliberation model.')
    parser.add_argument('modelpath', help='The model path, default is %(default)s')
    parser.add_argument('--log', action='store', dest='logfile', default=None, metavar='FILE',
                        help='The logging filename, default is %(default)s')
    parser.add_argument('--action', action='store', dest='action', default='a',
                        help='Test action (string contains "a", "p" and "r"), default is %(default)s')
    parser.add_argument('--bs', action='store', dest='valid_batch_size', default=80, type=int, metavar='N',
                        help='Batch size, default is %(default)s')
    parser.add_argument('--start', action='store', dest='start_idx', default=10000, type=int, metavar='N',
                        help='Start iteration of model, default is %(default)s')
    parser.add_argument('--step', action='store', dest='step_idx', default=10000, type=int, metavar='N',
                        help='Step iteration of model, default is %(default)s')
    parser.add_argument('--end', action='store', dest='end_idx', default=10000, type=int, metavar='N',
                        help='End iteration of model, default is %(default)s')
    parser.add_argument('-k', action='store', dest='k', default=1, type=int, metavar='N',
                        help='Get the top-k of word probabilities, default is %(default)s')

    args = parser.parse_args()

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
        k=args.k,
    )


if __name__ == '__main__':
    main()
