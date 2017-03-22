@rem Translate
@rem %1: model file; %2: target output directory
python translate.py %1 data/dic/filtered_dic_en-fr.en.pkl data/dic/filtered_dic_en-fr.fr.pkl data/test/test_en-fr.en.tok translated/%2/output.tok -p 1

@rem calculate multi BLEU score
perl multi-bleu.perl data/test/test_en-fr.fr.tok < translated/%2/output.tok
