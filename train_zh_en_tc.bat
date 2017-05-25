@rem Train of zh-en (truecased).
python train_nmt.py %* --train1 zh-en.1.25M.zh --train2 tc_zh-en.1.25M.en ^
--valid1 Nist2003.chs.word.max50.snt --valid2 tc_Nist2003.enu.word.max50.snt ^
--dic1 tc_zh-en.1.25M.zh.pkl --dic2 tc_zh-en.1.25M.en.pkl ^
--small1 small_zh-en.1.25M.zh --small2 small_tc_zh-en.1.25M.en
