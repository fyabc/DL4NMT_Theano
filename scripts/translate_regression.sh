#! /bin/bash

# Usage: translate_regression.sh gpu_id n_enc n_dec residual_enc attention iteration n_processes

export THEANO_FLAGS=device=gpu$1
mkdir -p translated/complete
python translate.py /datadrive/v-yanfa/model/complete/e${2}d${3}_res${4}_att${5}.iter${6}.npz data/dic/filtered_dic_en-fr.en.pkl data/dic/filtered_dic_en-fr.fr.pkl data/test/test_en-fr.en.tok translated/complete/e${2}d${3}_res${4}_att${5}_iter${6}.tok -k 4 -p ${7}
perl multi-bleu.perl data/test/test_en-fr.fr.tok < translated/complete/e${2}d${3}_res${4}_att${5}_iter${6}.tok

