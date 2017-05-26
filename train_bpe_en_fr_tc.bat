@rem Train BPE of en-fr (truecase).
python train_nmt.py %* --train1 tc_en-fr_en.tok.bpe.32000 --train2 tc_en-fr_fr.tok.bpe.32000 ^
--valid1 tc_dev_en-fr_en.tok.bpe.32000 --valid2 tc_dev_en-fr_fr.tok.bpe.32000 ^
--dic1 tc_en-fr_en.tok.bpe.32000.pkl --dic2 tc_en-fr_fr.tok.bpe.32000.pkl ^
--small1 tc_small_en-fr_en.tok.bpe.32000 --small2 tc_small_en-fr_fr.tok.bpe.32000
