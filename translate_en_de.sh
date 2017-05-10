#!/usr/bin/env bash

# Usage: translate_en_de.bat gpu_id n_enc n_dec residual_enc attention iteration n_processes [postfix]
export THEANO_FLAGS=device=gpu$1,floatX=float32
mkdir -p translated/complete
python translate.py /datadrive/v-yanfa/model/complete/e${2}d${3}_res${4}_att${5}${8}.iter${6}.npz data/dic/en-de.en.pkl data/dic/en-de.de.pkl data/test/newstest2014-deen-src.en.tok translated/complete/e${2}d${3}_res${4}_att${5}${8}_iter${6}.tok -p ${7} -k 4

# calculate multi BLEU score
perl multi-bleu.perl data/test/newstest2014-deen-ref.de.tok < translated/complete/e${2}d${3}_res${4}_att${5}${8}_iter${6}.tok
