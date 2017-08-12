#! /usr/bin/python
# -*- coding: utf-8 -*-

from libs.utility.trainer_utils import get_parser, parse_options, run


def main():
    parser = get_parser()

    options = parse_options(parser)

    run(options)


if __name__ == '__main__':
    main()
