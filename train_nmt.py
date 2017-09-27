import argparse
import sys
import os

from constants import Datasets
from gpu_manager import get_gpu_usage


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-R', action="store_false", default=True, dest='reload',
                        help='Reload old model, default to True, set to False')
    parser.add_argument('-d', action='store_true', default=False, dest='dump_before_train',
                        help='Dump before train default to False, set to True')
    parser.add_argument('--lr', action="store", metavar="learning_rate", dest="learning_rate", type=float, default=1.0,
                        help='Start learning rate, default is %(default)s')
    parser.add_argument('--optimizer', action='store', default='adadelta')
    parser.add_argument('--plot', action='store', default=None,
                        help='Plot filename, default is None (not plot) (deprecated).')
    parser.add_argument('--save_freq', action='store', default=5000, type=int, dest='save_freq',
                        help='Model save frequency, default is %(default)s')
    parser.add_argument('--valid_freq', action='store', default=500, type=int, dest='valid_freq',
                        help='Model validate frequency, default is %(default)s')
    parser.add_argument('--bleu_freq', action='store', default=5000, type=int, dest='bleu_freq',
                        help='Model BLEU test frequency, default is %(default)s')
    parser.add_argument('--bleu_start_id', action='store', default=0, type=int, dest='bleu_start_id',
                        help='Model BLEU test start from, default is %(default)s')
    parser.add_argument('--dim', action='store', default=512, type=int, dest='dim',
                        help='Dim of hidden units, default is %(default)s')
    parser.add_argument('--bs', action='store', default=32, type=int, dest='batch_size',
                        help='Train batch size, default is %(default)s')
    parser.add_argument('--dim_word', action='store', default=512, type=int, dest='dim_word',
                        help='Dim of word embedding, default is %(default)s')
    parser.add_argument('--maxlen', action='store', default=64, type=int, dest='maxlen',
                        help='Max sentence length, default is %(default)s')
    parser.add_argument('-S', action='store_false', default=True, dest='shuffle',
                        help='Shuffle data per epoch, default is True, set to False')
    parser.add_argument('--train1', action='store', metavar='filename', dest='train1', type=str,
                        default='filtered_en-fr.en',
                        help='Source train file, default is %(default)s')
    parser.add_argument('--train2', action='store', metavar='filename', dest='train2', type=str,
                        default='filtered_en-fr.fr',
                        help='Target train file, default is %(default)s')
    parser.add_argument('--small1', action='store', metavar='filename', dest='small1', type=str,
                        default='small_en-fr.en',
                        help='Source small train file, default is %(default)s')
    parser.add_argument('--small2', action='store', metavar='filename', dest='small2', type=str,
                        default='small_en-fr.fr',
                        help='Target small train file, default is %(default)s')
    parser.add_argument('--valid1', action='store', metavar='filename', dest='valid1', type=str,
                        default='dev_en.tok',
                        help='Source valid file, default is %(default)s')
    parser.add_argument('--valid2', action='store', metavar='filename', dest='valid2', type=str,
                        default='dev_fr.tok',
                        help='Target valid file, default is %(default)s')
    parser.add_argument('--dic1', action='store', metavar='filename', dest='dic1', type=str,
                        default='filtered_dic_en-fr.en.pkl',
                        help='Source dict file, default is %(default)s')
    parser.add_argument('--dic2', action='store', metavar='filename', dest='dic2', type=str,
                        default='filtered_dic_en-fr.fr.pkl',
                        help='Target dict file, default is %(default)s')
    parser.add_argument('--n_words_src', action='store', default=25000, type=int, dest='n_words_src',
                        help='Vocabularies in source side, default is %(default)s')
    parser.add_argument('--n_words_tgt', action='store', default=25000, type=int, dest='n_words_tgt',
                        help='Vocabularies in target side, default is %(default)s')

    parser.add_argument('model_file', nargs='?', default='model/baseline/baseline.npz',
                        help='Generated model file, default is "%(default)s"')
    parser.add_argument('pre_load_file', nargs='?', default='model/en2fr.iter160000.npz',
                        help='Pre-load model file, default is "%(default)s"')

    parser.add_argument('--enc', action='store', default=1, type=int, dest='n_encoder_layers',
                        help='Number of encoder layers, default is 1')
    parser.add_argument('--dec', action='store', default=1, type=int, dest='n_decoder_layers',
                        help='Number of decoder layers, default is 1')
    parser.add_argument('--conn', action='store', default=2, type=int, dest='connection_type',
                        help='Connection type, '
                             'default is 2 (bidirectional only in first layer, other layers are forward);'
                             '1 is divided bidirectional GRU')

    parser.add_argument('--unit', action='store', metavar='unit', dest='unit', type=str, default='lstm',
                        help='The unit type, default is "lstm", can be set to "gru" or "hmrnn".')

    parser.add_argument('--encoder_unit', action='store', metavar='unit', dest='encoder_unit', type=str, default=None,
                        help='The unit type of encoder, default is None (use type specified by --unit),'
                             ' can be set to "lstm", "gru" or "hmrnn".')
    parser.add_argument('--decoder_unit', action='store', metavar='unit', dest='decoder_unit', type=str, default=None,
                        help='The unit type of decoder, default is None (use type specified by --unit),'
                             ' can be set to "lstm", "gru" or "hmrnn".')

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
    parser.add_argument('--unit_size', action='store', default=2, type=int, dest='unit_size',
                        help='Number of unit size, default is %(default)s')
    parser.add_argument('--cond_unit_size', action='store', default=2, type=int, dest='cond_unit_size',
                        help='Number of cond unit size, default is %(default)s')
    parser.add_argument('--clip', action='store', metavar='clip', dest='clip', type=float, default=1.0,
                        help='Gradient clip rate, default is 1.0.')
    parser.add_argument('--manual', action='store_false', dest='auto', default=True,
                        help='Set dropout rate and grad clip rate manually.')
    parser.add_argument('--emb', action='store', metavar='filename', dest='given_embedding', type=str, default=None,
                        help='Given embedding model file, default is None')
    parser.add_argument('--lr_discount', action='store', metavar='freq', dest='lr_discount_freq', type=int,
                        default=80000, help='The learning rate discount frequency, default is 80000')

    parser.add_argument('--distribute', action = 'store', metavar ='type', dest = 'dist_type', type = str, default= None,
                        help = 'The distribution version, default is None (singe GPU mode), candiates are "mv", "mpi_reduce"')
    parser.add_argument('--syncbatch', action='store', metavar='batch', dest='sync_batch', type=int, default=1,
                        help='Sync batch frequency, default is 1')
    parser.add_argument('--recover_lr_iter', action='store', dest='dist_recover_lr', type = int, default=10000,
                        help='The mini-batch index to recover lrate in distributed mode, default is 10000.')
    parser.add_argument('--sync_models', action='store_true', dest='sync_models', default=False,
                        help='Sync grads, otherwise sync model parameters. Set to True')
    parser.add_argument('--all_att', action='store_true', dest='all_att', default=False,
                        help='Generate attention from all decoder layers, default is False, set to True')
    parser.add_argument('--avg_ctx', action='store_true', dest='avg_ctx', default=False,
                        help='Average all context vectors to get softmax, default is False, set to True')
    parser.add_argument('--dataset', action='store', dest='dataset', default='en-fr',
                        help='Dataset, default is "%(default)s"')
    parser.add_argument('--gpu_map_file', action='store', metavar='filename', dest='gpu_map_file', type=str,
                        default=None, help='The file containing gpu id mapping information, '
                                           'each line is in the form physical_gpu_id\\theano_id')
    parser.add_argument('--ft_patience', action='store', metavar='N', dest='fine_tune_patience', type=int, default=-1,
                        help='Fine tune patience, default is %(default)s, set 8 to enable it')

    parser.add_argument('--bottom_lstm', action='store_true', dest='bottom_lstm', default=False,
                        help='Use LSTM in the first layer of multiscale RNN structure, default is False, set to True')
    parser.add_argument('--use_mask', action='store_true', dest='use_mask', default=False,
                        help='Use masks rather than T.switch in multiscale RNN structure, default is False, set to True')
    parser.add_argument('--benefit_0_boundary', action='store_true', dest='benefit_0_boundary', default=False,
                        help="Mask lower layers' hidden state, default is False, set to True")
    parser.add_argument('--use_all_one_boundary', action='store_true', dest='use_all_one_boundary', default=False,
                        help="Use all one boundary in decoder, default is False, set to True")
    parser.add_argument('--use_enc_explicit_boundary', action='store_true', dest='use_enc_explicit_boundary', default=False,
                        help="Use explicit boundary in encoder, default is False, set to True")
    parser.add_argument('--use_dec_explicit_boundary', action='store_true', dest='use_dec_explicit_boundary', default=False,
                        help="Use explicit boundary in decoder, default is False, set to True")
    parser.add_argument('--boundary_type', action = 'store', metavar ='type', dest = 'boundary_type', type = str, default='ST',
                        help='Type of boundaries, default is "ST", candidates are "Gumbel_Softmax", "ST_Gumbel"')
    parser.add_argument('--hard_sigmoid_a', action='store', metavar='hard_sigmoid_a', dest='hard_sigmoid_a', type=float,
                        default=5.0, help='Hard_sigomoid_a, default is 5.0.')
    parser.add_argument('--temperature', action='store', metavar='temperature', dest='temperature', type=float, default=1.0,
                        help='Temperature for gumbel softmax, default is 1.0.')
    parser.add_argument('--enc_boundary_regularization', action='store', metavar='regularization_alpha',
                        dest='enc_boundary_regularization', type=float, default=0.0,
                        help='Boundary regularization for multiscale RNN at encoder, default is 0.0.')
    parser.add_argument('--dec_boundary_regularization', action='store', metavar='regularization_alpha',
                        dest='dec_boundary_regularization', type=float, default=0.0,
                        help='Boundary regularization for multiscale RNN at decoder, default is 0.0.')
    parser.add_argument('--layerwise_attention', action='store_true', dest='layerwise_attention', default=False,
                        help="Use layerwise attention, default is False, set to True")

    args = parser.parse_args()
    print args

    if args.residual_enc == 'None':
        args.residual_enc = None
    if args.residual_dec == 'None':
        args.residual_dec = None
    if args.dist_type != 'mv' and args.dist_type != 'mpi_reduce':
        args.dist_type = None

    # FIXME: Auto mode
    if args.auto:
        if args.n_encoder_layers <= 2:
            args.dropout = False
            args.clip = 1.0
        else:
            args.dropout = 0.1
            args.clip = 5.0

        if args.n_encoder_layers <= 1:
            args.residual_enc = None
        if args.n_decoder_layers <= 1:
            args.residual_dec = None
            args.attention_layer_id = 0

        args.cond_unit_size = args.unit_size

    if args.encoder_unit is None:
        args.encoder_unit = args.unit
    if args.decoder_unit is None:
        args.decoder_unit = args.unit

    # If dataset is not 'en-fr', old value of dataset options like 'args.train1' will be omitted
    if args.dataset != 'en-fr':
        args.train1, args.train2, \
        args.small1, args.small2, \
        args.valid1, args.valid2, valid3, \
        test1, test2, test3, \
        args.dic1, args.dic2 = \
            Datasets[args.dataset]

    print 'Command line arguments:'
    print args
    sys.stdout.flush()

     # Init multiverso or mpi and set theano flags.
    if args.dist_type == 'mv':
        try:
            import multiverso as mv
        except ImportError:
            import multiverso_ as mv

        # FIXME: This must before the import of theano!
        mv.init(sync=True)
        worker_id = mv.worker_id()
        workers_cnt = mv.workers_num()
    elif args.dist_type == 'mpi_reduce':
        from mpi4py import MPI
        communicator = MPI.COMM_WORLD
        worker_id = communicator.Get_rank()
        workers_cnt = communicator.Get_size()

    if args.dist_type:
        available_gpus = get_gpu_usage(workers_cnt)
        gpu_maps_info = {idx: idx for idx in available_gpus}
        if args.gpu_map_file:
            for line in open(args.gpu_map_file, 'r'):
                phy_id, theano_id = line.split()
                gpu_maps_info[int(phy_id)] = int(theano_id)
        theano_id = gpu_maps_info[available_gpus[worker_id]]
        print 'worker id:%d, using theano id:%d, physical id %d' % (worker_id, theano_id, available_gpus[worker_id])
        os.environ['THEANO_FLAGS'] = 'device=gpu{},floatX=float32'.format(theano_id)
        sys.stdout.flush()

    from nmt import train

    train(
        saveto=args.model_file,
        preload=args.pre_load_file,
        reload_=args.reload,
        dim_word=args.dim_word,
        dim=args.dim,
        decay_c=0.,
        clip_c=args.clip,
        lrate=args.learning_rate,
        optimizer=args.optimizer,
        patience=1000,
        maxlen=args.maxlen,
        batch_size=args.batch_size,
        valid_batch_size=128,
        dispFreq=1,
        saveFreq=args.save_freq,
        validFreq=args.valid_freq,  # Change it from 2500 to 5000 @ 6/11/2017.
        bleuFreq=args.bleu_freq,
        bleu_start_id=args.bleu_start_id,
        datasets=('./data/train/{}'.format(args.train1),
                  './data/train/{}'.format(args.train2)),
        valid_datasets=('./data/dev/{}'.format(args.valid1),
                        './data/dev/{}'.format(args.valid2),
                        './data/dev/{}'.format(valid3)),
        test_datasets=('./data/test/{}'.format(test1),
                       './data/test/{}'.format(test2),
                       './data/test/{}'.format(test3)),
        small_train_datasets=('./data/train/{}'.format(args.small1),
                              './data/train/{}'.format(args.small2)),
        vocab_filenames=('./data/dic/{}'.format(args.dic1),
                         './data/dic/{}'.format(args.dic2)),
        task=args.dataset,
        use_dropout=args.dropout,
        overwrite=False,

        # n_words=30000,
        # n_words_src=30000,
        n_words=args.n_words_tgt,
        n_words_src=args.n_words_src,

        # Options from v-yanfa
        dump_before_train=args.dump_before_train,
        plot_graph=args.plot,
        lr_discount_freq=args.lr_discount_freq,

        n_encoder_layers=args.n_encoder_layers,
        n_decoder_layers=args.n_decoder_layers,
        encoder_many_bidirectional=args.connection_type == 1,

        attention_layer_id=args.attention_layer_id,
        unit=args.unit,
        encoder_unit=args.encoder_unit,
        decoder_unit=args.decoder_unit,
        residual_enc=args.residual_enc,
        residual_dec=args.residual_dec,
        use_zigzag=args.use_zigzag,
        given_embedding=args.given_embedding,

        unit_size=args.unit_size,
        cond_unit_size=args.cond_unit_size,

        given_imm=True,
        dump_imm=True,
        shuffle_data=args.shuffle,

        decoder_all_attention=args.all_att,
        average_context=args.avg_ctx,

        dist_type=args.dist_type,
        sync_batch=args.sync_batch,
        dist_recover_lr_iter = args.dist_recover_lr,
        sync_models = args.sync_models,

        fine_tune_patience=args.fine_tune_patience,
        bottom_lstm=args.bottom_lstm,
        use_mask=args.use_mask,
        boundary_type_str=args.boundary_type,
        hard_sigmoid_a_schedule=args.hard_sigmoid_a,
        temperature_schedule=args.temperature,
        benefit_0_boundary=args.benefit_0_boundary,
        use_all_one_boundary=args.use_all_one_boundary,
        use_enc_explicit_boundary=args.use_enc_explicit_boundary,
        use_dec_explicit_boundary=args.use_dec_explicit_boundary,
        enc_boundary_regularization=args.enc_boundary_regularization,
        dec_boundary_regularization=args.dec_boundary_regularization,
        layerwise_attention=args.layerwise_attention,
    )


if __name__ == '__main__':
    main()
