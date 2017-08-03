#! /usr/bin/python
# -*- coding: utf-8 -*-

from libs.utility.trainer_utils import get_parser, parse_options


def main():
    parser = get_parser()

    options = parse_options(parser)


if __name__ == '__main__':
    main()
