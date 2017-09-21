#use this script to average several dumped models

import argparse

import numpy as np
import os
from collections import OrderedDict

def main(args=None):
    parser = argparse.ArgumentParser(description='Average models.')
    parser.add_argument('model_prefix', nargs='?', default='model/complete/e4d4_enfrlarge_reslayer_wise_att1_bpe_tc_lr0.1.npz',
                        help='The prefix of nmt model path, default is "%(default)s"')
    parser.add_argument('--start', action="store", metavar="index", dest="start", type=int, default=1,
                        help='The starting index of saved model to test, default is %(default)s')
    parser.add_argument('--end', action="store", metavar="index", dest="end", type=int, default=10,
                        help='The ending index of saved model to test, default is %(default)s')
    parser.add_argument('--gap', action="store", metavar="index", dest="interval", type=int, default=10000,
                        help='The interval between two consecutive tested models\' indexes, default is %(default)s')

    args = parser.parse_args(args)
    model_file = '%s.iter%d.npz' % (os.path.splitext(args.model_prefix)[0], args.start * args.interval)

    params_sum = np.load(model_file)
    params_dic = OrderedDict()
    for key, value in params_sum.iteritems():
        params_dic[key] = value

    Wemb = params_sum['Wemb']
    print 'w_emb_sum %.4f' % ( Wemb.sum())

    for idx in xrange(args.start + 1, args.end + 1):
        model_file = '%s.iter%d.npz' % (os.path.splitext(args.model_prefix)[0], idx * args.interval)
        new_params = np.load(model_file)
        for key, value in params_dic.iteritems():
            value += new_params[key]

    for key, value in params_dic.iteritems():
        value /= (args.end - args.start + 1)

    save_model_file_name = '%s.ave_s%d_e%d_i%d.npz' % (os.path.splitext(args.model_prefix)[0], args.start * args.interval,
                                                       args.end * args.interval, args.end - args.start + 1)
    np.savez(save_model_file_name, **params_dic)


if __name__ == '__main__':
    main()
