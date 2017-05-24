#! /bin/bash

# Usage: translate_bpe_en_de2_single.sh gpu_id n_enc n_dec residual_enc attention iteration [other]

export THEANO_FLAGS=device=gpu$1,floatX=float32
mkdir -p translated/complete

model_name=e${2}d${3}_res${4}_att${5}${7}
output_file=translated/complete/${model_name}_iter${6}.tok

python translate_single.py //gcr/Scratch/RR1/v-yanfa/SelectiveTrain/model/complete/${model_name}.iter${6}.npz data/dic/vocab.bpe.32000.pkl data/dic/vocab.bpe.32000.pkl data/test/newstest2014.tok.bpe.32000.en ${output_file} -k 4

cat ${output_file} | sed -r 's/(@@ )|(@@ ?$)//g' > ${output_file}.bpe

perl multi-bleu.perl data/test/test_en-de.de.tok < ${output_file}.bpe

