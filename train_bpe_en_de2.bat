@rem Train BPE of en-de 2 (seq2seq).
python train_nmt.py %* --train1 train.tok.clean.bpe.32000.en --train2 train.tok.clean.bpe.32000.de ^
--valid1 newstest2013.tok.bpe.32000.en --valid2 newstest2013.tok.bpe.32000.de ^
--dic1 vocab.bpe.32000.pkl --dic2 vocab.bpe.32000.pkl ^
--small1 small_train.tok.clean.bpe.32000.en --small2 small_train.tok.clean.bpe.32000.de
