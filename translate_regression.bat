@rem Translate regression init model.
@rem %1: model name (without prefix and postfix); %2: regression iteration; %3: -p
python translate.py model/init/en2fr_init_%1_iter%2.iter160000.npz data/dic/filtered_dic_en-fr.en.pkl data/dic/filtered_dic_en-fr.fr.pkl data/test/test_en-fr.en.tok translated/init/output_init_%1_iter%2.tok -p %3 -k 4

@rem calculate multi BLEU score
perl multi-bleu.perl data/test/test_en-fr.fr.tok < translated/init/output_init_%1_iter%2.tok