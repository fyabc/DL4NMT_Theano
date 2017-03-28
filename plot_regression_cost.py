#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import sys
import matplotlib.pyplot as plt

__author__ = 'fyabc'


def main():
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
                valid_iterations.append(iterations[-1])
                valid_costs.append(words[5])

        plt.plot(iterations, costs)
        plt.plot(valid_iterations, valid_costs)

        plt.show()


if __name__ == '__main__':
    main()
