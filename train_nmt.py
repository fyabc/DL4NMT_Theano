import numpy
import os
import argparse

import numpy
import sys

from nmt import train


def main(job_id, params):
    print params
    validerr = train(saveto=params['model'][0],
                     preload=params['pre_model'],
                     reload_=params['reload'][0],
                     dim_word=params['dim_word'][0],
                     dim=params['dim'][0],
                     decay_c=params['decay-c'][0],
                     clip_c=params['clip-c'][0],
                     lrate=params['learning-rate'][0],
                     optimizer=params['optimizer'][0],
                     patience=1000,
                     maxlen=1000,
                     batch_size=80,
                     dispFreq=2500,
                     saveFreq=params['save_freq'][0],
                     datasets=[r'.\data\train\edit_dis_top1M.en',
                               r'.\data\train\edit_dis_top1M.fr'],
                     use_dropout=params['use-dropout'][0],
                     overwrite=False,
                     # picked_train_idxes_file=params['train_idx_file'],
                     n_words=params['n-words'][0],
                     n_words_src=params['n-words'][0],
                     sort_by_len=params['curr'],
                     convert_embedding=params['convert_embedding'],
                     dump_before_train=params['dump_before_train'],
                     plot_graph=params['plot_graph'],
                     )
    return validerr


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # reload
    parser.add_argument('-R', action="store_false", default=True, dest='reload',
                        help='Reload, default to True, set to False')
    parser.add_argument('-C', action="store_false", default=False, dest='convert_embedding',
                        help='Convert embedding, default to True, set to False')
    parser.add_argument('-d', action='store_true', default=False, dest='dump_before_train')
    parser.add_argument('--lr', action="store", metavar="learning_rate", dest="learning_rate", type=float, default=0.8)
    parser.add_argument('-curri', action="store_true", default=False)
    parser.add_argument('--optimizer', action='store', default='sgd')
    parser.add_argument('--plot', action='store', default=None,
                        help='Plot filename, default is None (not plot)')

    parser.add_argument('model_file', nargs='?', default='model/top1M/en2fr_top1M.npz',
                        help='Generated model file, default is "model/top1M/en2fr_top1M.npz"')
    # parser.add_argument('train_idx_file', nargs='?', type=str, default='')  # the subset indexes chosen
    parser.add_argument('pre_load_file', nargs='?', default='model/en2fr.iter160000.npz',
                        help='Pre-load model file, default is "model/en2fr.iter160000.npz"')

    # [NOTE]
    # default arguments in my experiment
    # reload = True
    # Optimizer = sgd
    # pre_load_file = 'model/en2fr.iter160000.npz'
    # learning-rate = 0.4 (0.8 / 2)
    # save_freq = 10000
    # lr /= 2 per 80000 iteration

    args = parser.parse_args()
    print args
    # if len(args.train_idx_file) != 0:
    #     assert os.path.exists(args.train_idx_file)
    sys.stdout.flush()

    main(0, {
        'model': [args.model_file],
        'dim_word': [620],
        'dim': [1000],
        'n-words': [30000],
        'optimizer': [args.optimizer],
        'decay-c': [0.],
        'clip-c': [1.],
        'use-dropout': [False],
        # 'learning-rate': [1.],
        'learning-rate': [args.learning_rate],
        'reload': [args.reload],
        # 'save_freq': [5000],
        'save_freq': [10000],
        # 'train_idx_file': args.train_idx_file,
        'curr': args.curri,
        'pre_model': args.pre_load_file,
        'convert_embedding': args.convert_embedding,
        'dump_before_train': args.dump_before_train,
        'plot_graph': args.plot,
    })
