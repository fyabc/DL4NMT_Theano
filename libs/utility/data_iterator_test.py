from data_iterator import TextIterator
import cPickle as pkl
import sys

dic_file = '../../data/dic/tc_enfr_large_bpe.vocab.pkl'
dic = pkl.load(open(dic_file, 'rb'))
r_dic = {v:k for (k,v) in dic.iteritems()}

if __name__ == '__main__':
    k = int(sys.argv[1])
    text_iterator_fren = TextIterator(
                '../../data/train/tc_train_enfr_large_bpe.fr_00', '../../data/train/tc_train_enfr_large_bpe.en_00',
                dic_file, dic_file,
                64,36000, 36000,100, k = k
            )

    n_sample = 0
    for i, (x, y) in enumerate(text_iterator_fren):
            n_sample += len(x)
            sys.stdout.write('\rbatch:%d, n_samples: %d' % (i, n_sample))
