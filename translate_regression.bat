@rem Translate regression init model.
@rem %1: model name ; %2: output name; %3: -p
python translate.py %1 data/dic/filtered_dic_en-fr.en.pkl data/dic/filtered_dic_en-fr.fr.pkl data/test/test_en-fr.en.tok translated/%2 -p %3 -k 4

@rem calculate multi BLEU score
perl multi-bleu.perl data/test/test_en-fr.fr.tok < translated/%2