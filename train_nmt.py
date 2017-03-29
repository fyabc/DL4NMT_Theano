import numpy
import os
import argparse

import numpy
import sys

from nmt import train


def main(job_id, args):
    validerr = train(
        saveto=args.model_file,
        preload=args.pre_load_file,
        reload_=args.reload,
        dim_word=620,
        dim=1000,
        decay_c=0.,
        clip_c=1.,
        lrate=args.learning_rate,
        optimizer=args.optimizer,
        patience=1000,
        maxlen=1000,
        batch_size=80,
        dispFreq=2500,
        saveFreq=10000,
        datasets=[r'.\data\train\filtered_en-fr.en',
                  r'.\data\train\filtered_en-fr.fr'],
        use_dropout=False,
        overwrite=False,
        n_words=30000,
        n_words_src=30000,
        sort_by_len=args.curri,

        # Options from v-yanfa
        convert_embedding=args.pre_load_file,
        dump_before_train=args.dump_before_train,
        plot_graph=args.plot,

        n_encoder_layers=args.n_encoder_layers,
        n_decoder_layers=args.n_decoder_layers,
        encoder_many_bidirectional=args.connection_type == 1,
    )
    return validerr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # FIXME: If reload set to True, the model will be set as this:
    # Encoder/Decoder layer 0: copy
    # Encoder/Decoder layer 1 ~ last: set weights to identity
    parser.add_argument('-R', action="store_false", default=True, dest='reload',
                        help='Reload, default to True, set to False')
    parser.add_argument('-c', action="store_true", default=False, dest='convert_embedding',
                        help='Convert embedding, default to False, set to True (deprecated)')
    parser.add_argument('-d', action='store_true', default=False, dest='dump_before_train',
                        help='Dump before train default to False, set to True')
    parser.add_argument('--lr', action="store", metavar="learning_rate", dest="learning_rate", type=float, default=0.8)
    parser.add_argument('-curri', action="store_true", default=False)
    parser.add_argument('--optimizer', action='store', default='sgd')
    parser.add_argument('--plot', action='store', default=None,
                        help='Plot filename, default is None (not plot)')

    parser.add_argument('model_file', nargs='?', default='model/baseline/baseline.npz',
                        help='Generated model file, default is "model/baseline/baseline.npz"')
    # parser.add_argument('train_idx_file', nargs='?', type=str, default='')  # the subset indexes chosen
    parser.add_argument('pre_load_file', nargs='?', default='model/en2fr.iter160000.npz',
                        help='Pre-load model file, default is "model/en2fr.iter160000.npz"')

    parser.add_argument('--enc', action='store', default=1, type=int, dest='n_encoder_layers',
                        help='Number of encoder layers, default is 1')
    parser.add_argument('--dec', action='store', default=1, type=int, dest='n_decoder_layers',
                        help='Number of decoder layers, default is 1')
    parser.add_argument('--conn', action='store', default=1, type=int, dest='connection_type',
                        help='Connection type, default is 1 (divided bidirectional GRU);\n'
                             '2 is bidirectional only in first layer, other layers are forward')

    # [NOTE]
    # default arguments in my experiment
    # reload = True
    # Optimizer = sgd
    # pre_load_file = 'model/en2fr.iter160000.npz'
    # learning-rate = 0.4 (0.8 / 2)
    # save_freq = 10000
    # lr /= 2 per 80000 iteration

    args = parser.parse_args()
    print 'Command line arguments:'
    print args
    # if len(args.train_idx_file) != 0:
    #     assert os.path.exists(args.train_idx_file)
    sys.stdout.flush()

    main(0, args)
