@rem Train BPE of en-de.
python train_nmt.py %* --train1 en-de.en_0 --train2 en-de_de.0 ^
--valid1 dev_en.tok --valid2 dev_de.tok ^
--dic1 en-de.en.pkl --dic2 en-de.de.pkl ^
--small1 small_en-de.en_0 --small2 small_en-de_de.0
