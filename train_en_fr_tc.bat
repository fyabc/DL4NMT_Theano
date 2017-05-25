@rem Train of en-fr (truecased).
python train_nmt.py %* --train1 tc_filtered_en-fr.en --train2 tc_filtered_en-fr.fr ^
--valid1 tc_dev_en.tok --valid2 tc_dev_fr.tok ^
--dic1 tc_filtered_dic_en-fr.en.pkl --dic2 tc_filtered_dic_en-fr.fr.pkl ^
--small1 small_tc_en-fr.en --small2 small_tc_en-fr.fr
