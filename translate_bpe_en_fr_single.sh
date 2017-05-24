#! /bin/bash

# Usage: translate_bpe_en_fr_single.sh gpu_id n_enc n_dec residual_enc attention iteration [other]

export THEANO_FLAGS=device=gpu$1,floatX=float32
mkdir -p translated/complete

model_name=e${2}d${3}_res${4}_att${5}${7}
output_file=translated/complete/${model_name}_iter${6}.tok

python translate.py /datadrive/v-yanfa/model/complete/${model_name}.iter${6}.npz data/dic/en-fr_vocab.bpe.32000.pkl data/dic/en-fr_vocab.bpe.32000.pkl data/test/test_en-fr.en.tok.bpe.32000 ${output_file} -k 4

cat ${output_file} | sed -r 's/(@@ )|(@@ ?$)//g' > ${output_file}.bpe

perl multi-bleu.perl data/test/test_en-fr.fr.tok < ${output_file}.bpe

