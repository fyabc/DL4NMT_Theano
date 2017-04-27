#! /bin/bash

# Usage: translate_single_bpe.sh gpu_id n_enc n_dec residual_enc attention iteration [other]

export THEANO_FLAGS=device=gpu$1,floatX=float32
mkdir -p translated/complete

model_name=e${2}d${3}_res${4}_att${5}${7}
output_file=translated/complete/${model_name}_iter${6}.tok

python translate_single.py /datadrive/v-yanfa/model/complete/${model_name}.iter${6}.npz data/dic/filtered_dic_en-fr.en.pkl data/dic/filtered_dic_en-fr.fr.pkl data/test/test_en-fr.en.tok ${output_file} -k 4

cat ${output_file} | sed -r 's/(@@ )|(@@ ?$)//g' > ${output_file}

perl multi-bleu.perl data/test/test_en-fr.fr.tok < ${output_file}
