@rem Usage: translate_en_de.bat n_enc n_dec residual_enc attention iteration n_processes [postfix]
python translate.py //gcr/Scratch/RR1/v-yanfa/SelectiveTrain/model/complete/e%1d%2_res%3_att%4%7.iter%5.npz ^
data/dic/en-de.en.pkl data/dic/en-de.de.pkl data/test/newstest2014-deen-src.en.tok ^
translated/complete/e%1d%2_res%3_att%4%7_iter%5.tok -p %6 -k 4

@rem calculate multi BLEU score
perl multi-bleu.perl data/test/newstest2014-deen-ref.de.tok < translated/complete/e%1d%2_res%3_att%4%7_iter%5.tok
