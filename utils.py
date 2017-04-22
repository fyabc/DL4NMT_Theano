#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

from collections import OrderedDict
import warnings
import os
import cPickle as pkl
from pprint import pprint
import re
import errno

import theano
import theano.tensor as tensor
import numpy as np

from constants import fX
from multiverso.theano_ext import sharedvar


_fp_log = None


def set_logging_file(logging_filename):
    path, filename = os.path.split(logging_filename)

    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            print('ERROR when creating the logging file: {}'.format(e.message))

    global _fp_log
    _fp_log = open(logging_filename, 'w')


def get_logging_file():
    return _fp_log


def message(*args, **kwargs):
    """Print message both to logging file and stdout."""
    print(*args, **kwargs)

    if _fp_log is not None:
        print(*args, **kwargs, file=_fp_log)


def log(*args, **kwargs):
    """Print message to logging file."""
    if _fp_log is not None:
        print(*args, **kwargs, file=_fp_log)


def close_logging_file():
    if _fp_log is not None:
        _fp_log.close()


def zipp(params, tparams):
    """Push parameters to Theano shared variables"""
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)


def unzip(zipped):
    """Pull parameters from Theano shared variables"""
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params


def itemlist(tparams):
    """Get the list of parameters: Note that tparams must be OrderedDict"""
    return [vv for kk, vv in tparams.iteritems()]


def _p(*args, **kwargs):
    """Make prefix-appended name"""

    # FIXME: To be compatible with old model, when the layer id is 0 and open 'layer_id_compatible', omit the layer id.
    layer_id_compatible = kwargs.pop('layer_id_compatible', True)
    if layer_id_compatible and args[-1] == 0:
        args = args[:-1]

    return '_'.join(str(arg) for arg in args)


# These parameters should be duplicated for multiverso.
dup_shared_var_list = ['decoder_c_tt']
dup_size = 100


def init_tparams(params, given_tparams=None, given_dup_tparams=None, sync=False):
    """Initialize Theano shared variables according to the initial parameters"""

    tparams = OrderedDict() if given_tparams is None else given_tparams
    dup_tparams = OrderedDict() if given_dup_tparams is None else given_dup_tparams

    if not sync:
        for kk, pp in params.iteritems():
            tparams[kk] = theano.shared(params[kk], name=kk)
    else:
        for kk, pp in params.iteritems():
            if any(kk.startswith(var) for var in dup_shared_var_list):
                tparams[kk] = theano.shared(params[kk], name=kk)
                dup_tparams[kk] = sharedvar.mv_shared(value=np.ones(dup_size) * params[kk][0], name=kk, borrow=False)
            else:
                tparams[kk] = sharedvar.mv_shared(value=params[kk], name=kk, borrow=False)
    return tparams, dup_tparams


def sync_tparams(tparams, dup_tparams):
    for kk, vv in dup_tparams.iteritems():
        vv.set_value(np.ones(dup_size) * tparams[kk].get_value()[0])
    sharedvar.sync_all_mv_shared_vars()
    for kk, vv in dup_tparams.iteritems():
        tparams[kk].set_value(np.array([vv.get_value()[0]], dtype=fX).reshape((1,)))


def load_params(path, params):
    """Load parameters

    :param path: Path of old parameters.
    :param params: New parameters to be updated.
    """

    old_params = np.load(path)
    for key, value in params.iteritems():
        if key not in old_params:
            warnings.warn('{} is not in the archive'.format(key))
            continue
        params[key] = old_params[key]

    return params


def load_embedding(params, embedding_model_file, emb_keys=('Wemb', 'Wemb_dec')):
    embedding_model = np.load(embedding_model_file)

    for key in emb_keys:
        params[key] = embedding_model[key]

    return params


# some utilities
def orthogonal_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype('float32')


def normal_weight(nin, nout=None, scale=0.01, orthogonal=True):
    if nout is None:
        nout = nin
    if nout == nin and orthogonal:
        W = orthogonal_weight(nin)
    else:
        W = np.random.randn(nin, nout)
        u, s, v = np.linalg.svd(W)
        if nin > nout:
            W = u[:, :nout]
        else:
            W = v[:nin, :]
    return W.astype('float32')


def uniform_weight(nin, nout=None, scale=0.01):
    if nout is None:
        nout = nin
    return np.random.uniform(-1. * scale, 1. * scale, (nin, nout)).astype('float32')


def concatenate(tensor_list, axis=0):
    """
    Alternative implementation of `theano.tensor.concatenate`.
    This function does exactly the same thing, but contrary to Theano's own
    implementation, the gradient is implemented on the GPU.
    Backpropagating through `theano.tensor.concatenate` yields slowdowns
    because the inverse operation (splitting) needs to be done on the CPU.
    This implementation does not have that problem.
    :usage:
        >>> x, y = theano.tensor.matrices('x', 'y')
        >>> c = concatenate([x, y], axis=1)
    :parameters:
        - tensor_list : list
            list of Theano tensor expressions that should be concatenated.
        - axis : int
            the tensors will be joined along this axis.
    :returns:
        - out : tensor
            the concatenated tensor expression.
    """
    concat_size = sum(tt.shape[axis] for tt in tensor_list)

    output_shape = ()
    for k in range(axis):
        output_shape += (tensor_list[0].shape[k],)
    output_shape += (concat_size,)
    for k in range(axis + 1, tensor_list[0].ndim):
        output_shape += (tensor_list[0].shape[k],)

    out = tensor.zeros(output_shape)
    offset = 0
    for tt in tensor_list:
        indices = ()
        for k in range(axis):
            indices += (slice(None),)
        indices += (slice(offset, offset + tt.shape[axis]),)
        for k in range(axis + 1, tensor_list[0].ndim):
            indices += (slice(None),)

        out = tensor.set_subtensor(out[indices], tt)
        offset += tt.shape[axis]

    return out


def average(l):
    if not l:
        return 0.0
    return sum(l) / len(l)


def apply_gradient_clipping(clip_c, grads):
    g2 = 0.
    if clip_c > 0.:
        for g in grads:
            g2 += (g ** 2).sum()
        new_grads = []
        for g in grads:
            new_grads.append(tensor.switch(g2 > (clip_c ** 2),
                                           g / tensor.sqrt(g2) * clip_c,
                                           g))
        grads = new_grads
    return grads, g2


def l2_regularization(cost, tparams, decay_c):
    """Apply L2 regularization on weights."""
    if decay_c > 0.:
        decay_c = theano.shared(np.float32(decay_c), name='decay_c')
        weight_decay = 0.
        for kk, vv in tparams.iteritems():
            weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay
    return cost


def regularize_alpha_weights(cost, alpha_c, model_options, x_mask, y_mask, opt_ret):
    """Regularize the alpha weights."""
    if alpha_c > 0. and not model_options['decoder'].endswith('simple'):
        alpha_c = theano.shared(np.float32(alpha_c), name='alpha_c')
        alpha_reg = alpha_c * (
            (tensor.cast(y_mask.sum(0) // x_mask.sum(0), fX)[:, None] -
             opt_ret['dec_alphas'].sum(0)) ** 2).sum(1).mean()
        cost += alpha_reg
    return cost


# batch preparation
def prepare_data(seqs_x, seqs_y, maxlen=None):
    # x: a list of sentences
    lengths_x = [len(s) for s in seqs_x]
    lengths_y = [len(s) for s in seqs_y]

    if maxlen is not None:
        new_seqs_x = []
        new_seqs_y = []
        new_lengths_x = []
        new_lengths_y = []
        for l_x, s_x, l_y, s_y in zip(lengths_x, seqs_x, lengths_y, seqs_y):
            if l_x < maxlen and l_y < maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
                new_seqs_y.append(s_y)
                new_lengths_y.append(l_y)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x
        lengths_y = new_lengths_y
        seqs_y = new_seqs_y

        if len(lengths_x) < 1 or len(lengths_y) < 1:
            return None, None, None, None

    n_samples = len(seqs_x)
    maxlen_x = np.max(lengths_x) + 1
    maxlen_y = np.max(lengths_y) + 1

    x = np.zeros((maxlen_x, n_samples)).astype('int64')
    y = np.zeros((maxlen_y, n_samples)).astype('int64')
    x_mask = np.zeros((maxlen_x, n_samples)).astype('float32')
    y_mask = np.zeros((maxlen_y, n_samples)).astype('float32')
    for idx, [s_x, s_y] in enumerate(zip(seqs_x, seqs_y)):
        x[:lengths_x[idx], idx] = s_x
        x_mask[:lengths_x[idx] + 1, idx] = 1.
        y[:lengths_y[idx], idx] = s_y
        y_mask[:lengths_y[idx] + 1, idx] = 1.

    return x, x_mask, y, y_mask


def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """
    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
        minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if minibatch_start != n:
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return list(enumerate(minibatches))


# Debug utilities
def print_params(params, exit_=False):
    total_parameters = 0

    print('Model Parameters:')
    for k, v in params.iteritems():
        print('  >', k, v.shape, v.dtype)
        total_parameters += v.size
    print('Total parameters of the network: {}'.format(total_parameters))
    print('Model Parameters Done')

    if exit_:
        exit(0)


def load_options(options, reload_=None, preload=None, print_options=True):
    """Reload options."""

    reload_ = options['reload_'] if reload_ is None else reload_
    preload = options['preload'] if preload is None else preload

    if reload_ and os.path.exists(preload):
        print('Reloading model options')
        with open('{}.pkl'.format(preload), 'rb') as f:
            # model_options = pkl.load(f)
            # FIXME: Update the option instead of replace it
            options.update(pkl.load(f))

        # Remain reload_ and preload
        options['reload_'] = reload_
        options['preload'] = preload

    if print_options:
        message('Model options:')
        pprint(options)
        pprint(options, stream=get_logging_file())
        message()


def save_options(options, iteration, saveto=None):
    saveto = options['saveto'] if saveto is None else saveto

    save_filename = '{}.iter{}.npz.pkl'.format(os.path.splitext(saveto)[0], iteration)
    with open(save_filename, 'wb') as f:
        pkl.dump(options, f)


def search_start_uidx(reload_, preload):
    if not reload_:
        return 0

    m = re.search('.+iter(\d+?)\.npz', preload)
    if m:
        return int(m.group(1))
    else:
        return 0


def make_f_train(f_grad_shared, f_update):
    def f_train(x, x_mask, y, y_mask, lr):
        cost = f_grad_shared(x, x_mask, y, y_mask)

        f_update(lr)

        return cost

    return f_train


__all__ = [
    'set_logging_file',
    'get_logging_file',
    'message',
    'log',
    'close_logging_file',
    'zipp',
    'unzip',
    'itemlist',
    '_p',
    'init_tparams',
    'sync_tparams',
    'load_params',
    'load_embedding',
    'orthogonal_weight',
    'normal_weight',
    'uniform_weight',
    'concatenate',
    'average',
    'apply_gradient_clipping',
    'l2_regularization',
    'regularize_alpha_weights',
    'prepare_data',
    'get_minibatches_idx',
    'print_params',
    'load_options',
    'save_options',
    'search_start_uidx',
    'make_f_train',
]
