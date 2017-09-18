@rem Translate
@rem %1: target output directory %2: iteration
python translate.py model/%1/en2fr_%1.iter%2.npz data/dic/filtered_dic_en-fr.en.pkl data/dic/filtered_dic_en-fr.fr.pkl data/test/test_en-fr.en.tok translated/%1/output_iter%2.tok -p 1 -k 12

@rem calculate multi BLEU score
perl multi-bleu.perl data/test/test_en-fr.fr.tok < translated/%1/output_iter%2.tok
