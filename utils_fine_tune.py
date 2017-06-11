#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function

import os
import re
import subprocess

__author__ = 'fyabc'


def get_bleu(ref_file, hyp_file):
    pl_output = subprocess.Popen(
        'perl multi-bleu.perl {} < {}\n'.format(ref_file, hyp_file), shell=True,
        stdout=subprocess.PIPE, stderr=open(os.devnull, 'w')).stdout.read()

    contents = pl_output.split(',')
    if len(contents) == 0:
        return 0.0
    var = contents[0].split(" = ")
    if len(var) <= 1:
        return 0.0
    BLEU = var[1]

    return float(BLEU)


def de_bpe(input_str):
    return re.sub(r'(@@ )|(@@ ?$)', '', input_str)


def translate_dev_get_bleu(model, f_init, f_next, trng, dataset):
    # todo: translate the model
    return 0.0


__all__ = [
    'get_bleu',
    'de_bpe',
    'translate_dev_get_bleu',
]
