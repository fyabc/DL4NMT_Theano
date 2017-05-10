#!/usr/bin/env bash

# Usage: translate_en_de.bat n_enc n_dec residual_enc attention iteration n_processes [postfix]
python translate.py /datadrive/v-yanfa/model/complete/e${1}d${2}_res${3}_att${4}${7}.iter${5}.npz data/dic/en-de.en.pkl data/dic/en-de.de.pkl data/test/newstest2014-deen-src.en.tok translated/complete/e${1}d${2}_res${3}_att${4}${7}_iter${5}.tok -p ${6} -k 4

# calculate multi BLEU score
perl multi-bleu.perl data/test/newstest2014-deen-ref.de.tok < translated/complete/e${1}d${2}_res${3}_att${4}${7}_iter${5}.tok
