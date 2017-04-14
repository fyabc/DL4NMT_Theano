import argparse
import sys

from nmt import train


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-R', action="store_false", default=True, dest='reload',
                        help='Reload old model, default to True, set to False')
    parser.add_argument('-d', action='store_true', default=False, dest='dump_before_train',
                        help='Dump before train default to False, set to True')
    parser.add_argument('--lr', action="store", metavar="learning_rate", dest="learning_rate", type=float, default=1.0,
                        help='Start learning rate, default is 1.0')
    parser.add_argument('--optimizer', action='store', default='adadelta')
    parser.add_argument('--plot', action='store', default=None,
                        help='Plot filename, default is None (not plot) (deprecated).')

    parser.add_argument('model_file', nargs='?', default='model/baseline/baseline.npz',
                        help='Generated model file, default is "model/baseline/baseline.npz"')
    parser.add_argument('pre_load_file', nargs='?', default='model/en2fr.iter160000.npz',
                        help='Pre-load model file, default is "model/en2fr.iter160000.npz"')

    parser.add_argument('--enc', action='store', default=1, type=int, dest='n_encoder_layers',
                        help='Number of encoder layers, default is 1')
    parser.add_argument('--dec', action='store', default=1, type=int, dest='n_decoder_layers',
                        help='Number of decoder layers, default is 1')
    parser.add_argument('--conn', action='store', default=2, type=int, dest='connection_type',
                        help='Connection type, '
                             'default is 2 (bidirectional only in first layer, other layers are forward);'
                             '1 is divided bidirectional GRU')
    parser.add_argument('--unit', action='store', metavar='unit', dest='unit', type=str, default='lstm',
                        help='The unit type, default is "lstm", can be set to "gru".')
    parser.add_argument('--attention', action='store', metavar='index', dest='attention_layer_id', type=int, default=0,
                        help='Attention layer index, default is 0')
    parser.add_argument('--residual_enc', action='store', metavar='type', dest='residual_enc', type=str, default=None,
                        help='Residual connection of encoder, default is None, candidates are "layer_wise", "last"')
    parser.add_argument('--residual_dec', action='store', metavar='type', dest='residual_dec', type=str,
                        default='layer_wise',
                        help='Residual connection of decoder, default is "layer_wise", candidates are None, "last"')
    parser.add_argument('-z', '--zigzag', action='store_false', default=True, dest='use_zigzag',
                        help='Use zigzag in encoder, default is True, set to False')

    parser.add_argument('--dropout', action="store", metavar="dropout", dest="dropout", type=float, default=False,
                        help='Dropout rate, default is False (not use dropout)')
    parser.add_argument('--clip', action='store', metavar='clip', dest='clip', type=float, default=1.0,
                        help='Gradient clip rate, default is 1.0.')
    parser.add_argument('--manual', action='store_false', dest='auto', default=True,
                        help='Set dropout rate and grad clip rate manually.')

    args = parser.parse_args()

    # FIXME: Auto mode
    if args.auto:
        if args.n_encoder_layers <= 2:
            args.dropout = 0.1
            args.clip = 1.0
        else:
            args.dropout = False
            args.clip = 5.0

    print 'Command line arguments:'
    print args
    sys.stdout.flush()

    train(
        saveto=args.model_file,
        preload=args.pre_load_file,
        reload_=args.reload,
        dim_word=512,
        dim=512,
        decay_c=0.,
        clip_c=args.clip,
        lrate=args.learning_rate,
        optimizer=args.optimizer,
        patience=1000,
        maxlen=64,
        batch_size=128,
        valid_batch_size=128,
        dispFreq=1,
        saveFreq=10000,
        validFreq=2500,
        datasets=('./data/train/filtered_en-fr.en',
                  './data/train/filtered_en-fr.fr'),
        valid_datasets=('./data/dev/dev_en.tok',
                        './data/dev/dev_fr.tok'),
        small_train_datasets=('./data/train/small_en-fr.en',
                              './data/train/small_en-fr.fr'),
        use_dropout=args.dropout,
        overwrite=False,
        n_words=30000,
        n_words_src=30000,

        # Options from v-yanfa
        dump_before_train=args.dump_before_train,
        plot_graph=args.plot,

        n_encoder_layers=args.n_encoder_layers,
        n_decoder_layers=args.n_decoder_layers,
        encoder_many_bidirectional=args.connection_type == 1,

        attention_layer_id=args.attention_layer_id,
        unit=args.unit,
        residual_enc=args.residual_enc,
        residual_dec=args.residual_dec,
        use_zigzag=args.use_zigzag,
    )


if __name__ == '__main__':
    main()
