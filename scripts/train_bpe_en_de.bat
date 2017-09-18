@rem Train BPE of en-de.
python train_nmt.py %* --train1 en-de_en.tok.bpe.32000 --train2 en-de_de.tok.bpe.32000 ^
--valid1 dev_en-de_en.tok.bpe.32000 --valid2 dev_en-de_de.tok.bpe.32000 ^
--dic1 en-de_vocab.bpe.32000.pkl --dic2 en-de_vocab.bpe.32000.pkl ^
--small1 small_en-de_en.tok.bpe.32000 --small2 small_en-de_de.tok.bpe.32000
