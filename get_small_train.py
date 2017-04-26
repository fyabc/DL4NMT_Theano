#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function

import sys
import os
import random

__author__ = 'fyabc'


def main():
    input_filename = sys.argv[1]

    if len(sys.argv) >= 3:
        small_size = int(sys.argv[2])
    else:
        small_size = 10000

    with open(input_filename, 'r') as f_in:
        lines = list(f_in)

        selected_lines = random.sample(lines, small_size)

        head, tail = os.path.split(input_filename)
        output_filename = '{}{}small_{}'.format(head, '/' if head else '', tail)
        with open(output_filename, 'w') as f_out:
            for line in selected_lines:
                print(line, end='', file=f_out)

        print('Extract {} -> {}'.format(input_filename, output_filename))

if __name__ == '__main__':
    main()
