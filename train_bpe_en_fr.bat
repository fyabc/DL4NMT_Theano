@rem Train BPE of en-fr.
python train_nmt.py %* --train1 en-fr_en.tok.bpe.32000 --train2 en-fr_fr.tok.bpe.32000 ^
--valid1 dev_en-fr_en.tok.bpe.32000 --valid2 dev_en-fr_fr.tok.bpe.32000 ^
--dic1 en-fr_vocab.bpe.32000.pkl --dic2 en-fr_vocab.bpe.32000.pkl ^
--small1 small_en-fr_en.tok.bpe.32000 --small2 small_en-fr_fr.tok.bpe.32000
