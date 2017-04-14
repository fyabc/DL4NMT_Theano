#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import sys
import argparse

import matplotlib.pyplot as plt

__author__ = 'fyabc'


def average(l):
    if not l:
        return 0.0
    return sum(l) / len(l)


def plot(args):
    """Logging format:

    Epoch 0 Update 52 Cost 219.29276 G2 1483.47644 UD 2.78200 Time 127.66500 s
    """

    logging_file = sys.argv[1]

    with open(logging_file, 'r') as f:
        iterations = []
        costs = []

        valid_iterations = []
        valid_costs = []

        for line in f:
            if line.startswith('Epoch'):
                words = line.split()
                iterations.append(int(words[3]))
                costs.append(float(words[5]))
            elif line.startswith('Valid'):
                words = line.split()
                valid_iterations.append(iterations[-1] if iterations else 0)
                valid_costs.append(words[5])

        avg_costs = [average(costs[max(0, i - args.interval): i]) for i in xrange(len(costs))]

        plt.plot(iterations, avg_costs)
        plt.plot(valid_iterations, valid_costs)

        plt.show()


def main(args=None):
    parser = argparse.ArgumentParser(description='Plot cost curve.')
    parser.add_argument('filename', help='The logging filename')
    parser.add_argument('-a', '--average', action='store', metavar='interval', dest='interval', type=int, default=40,
                        help='The moving average interval, default is 40')

    args = parser.parse_args(args)

    plot(args)


if __name__ == '__main__':
    main()
