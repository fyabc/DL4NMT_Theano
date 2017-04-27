@rem Translate complete model.
@rem Usage: translate_complete.bat gpu_id n_enc n_dec residual_enc attention iteration thread_num [other]

set THEANO_FLAGS="device=gpu%1,floatX=float32"
set model_name=e%2d%3_res%4_att%5%8
set output_file=translated/complete/%model_name%_iter%6.tok

python translate.py \\gcr\Scratch\RR1\v-yanfa\SelectiveTrain\model\complete\%model_name%.iter%6.npz ^
data/dic/filtered_dic_en-fr.en.pkl data/dic/filtered_dic_en-fr.fr.pkl ^
data/test/test_en-fr.en.tok %output_file% -p %7 -k 4

perl multi-bleu.perl data/test/test_en-fr.fr.tok < %output_file%
