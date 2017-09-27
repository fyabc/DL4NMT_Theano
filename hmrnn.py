import numpy as np
import theano
from theano import tensor as T

from constants import fX, profile
from utils import _p, normal_weight, orthogonal_weight, concatenate
from theano.updates import OrderedUpdates

from theano.scalar.basic import UnaryScalarOp, same_out_nocomplex
from theano.tensor.elemwise import Elemwise
from theano.ifelse import ifelse

__author__ = 'zhuohan'

def prepare_explicit_boundary(data, idict):
    explicit_boundary = np.ones((data.shape[0], data.shape[1], 1), dtype=fX)
    for i in xrange(data.shape[0]):
        for j in xrange(data.shape[1]):
            if idict[data[i][j]][-2:] == '@@':
                explicit_boundary[i][j][0] = 0.0
    return explicit_boundary


def print_all(msg):
    return lambda x: theano.printing.Print(msg, attrs=['__str__', 'shape'])(x)

def _hmrnn_slice(_x, dim):
    """Slice function for HMRNN cell"""

    if _x.ndim == 3:
        return tuple([_x[:, :, n * dim:(n + 1) * dim] for n in xrange(4)]
                     + [_x[:, :, 4*dim: 4*dim+1]])
    return tuple([_x[:, n * dim:(n + 1) * dim] for n in xrange(4)]
                 + [_x[:, 4*dim: 4*dim+1]])


# Activations.

def tanh(x):
    return T.tanh(x)


def linear(x):
    return x


def bernoulli_gumbel_softmax(value, trng, noise_scale=1.0, temperature=1.0, eps=1e-3):
    U = trng.uniform(value.shape)
    U = T.log(U + eps) - T.log(1 - U + eps)
    return T.nnet.sigmoid((value + noise_scale * U) / temperature)


class Round3(UnaryScalarOp):
    def c_code(self, node, name, (x, ), (z, ), sub):
        return "%(z)s = round(%(x)s);" % locals()

    def grad(self, inputs, gout):
        (gz,) = gout
        return gz,


round3_scalar = Round3(same_out_nocomplex, name='round3')
round3 = Elemwise(round3_scalar)

def hard_sigmoid(x, a=5.0):
    return T.clip((a*x+1.)/2., 0, 1)

def dropout_layer(state_before, use_noise, trng, dropout_rate=0.5):
    """Dropout"""

    projection = T.switch(
        use_noise,
        state_before * trng.binomial(state_before.shape, p=(1. - dropout_rate), n=1,
                                     dtype=state_before.dtype),
        state_before * (1. - dropout_rate))
    return projection

def param_init_hmrnn(O, params, prefix='hmrnn', nin=None, dim=None, **kwargs):
    if nin is None:
        nin = O['dim_proj']
    if dim is None:
        dim = O['dim_proj']

    n_layers = kwargs.pop('n_layers', 2)

    # todo: add context support
    # todo: add support to arbitrary input dimension
    # context_dim = kwargs.pop('context_dim', None)

    params[_p(prefix, 'W0')] = np.concatenate([
        normal_weight(nin, dim),
        normal_weight(nin, dim),
        normal_weight(nin, dim),
        normal_weight(nin, dim),
        normal_weight(nin, 1)
    ], axis=1)

    params[_p(prefix, 'W')] = np.stack([
        np.concatenate([
            normal_weight(dim, dim),
            normal_weight(dim, dim),
            normal_weight(dim, dim),
            normal_weight(dim, dim),
            normal_weight(dim, 1)
        ], axis=1)
        for _ in xrange(n_layers - 1)
    ], axis=0)

    params[_p(prefix, 'U')] = np.stack([
        np.concatenate([
            orthogonal_weight(dim),
            orthogonal_weight(dim),
            orthogonal_weight(dim),
            orthogonal_weight(dim),
            normal_weight(dim, 1)
        ], axis=1)
        for _ in xrange(n_layers)
    ], axis=0)

    params[_p(prefix, 'Ut')] = np.stack([
        np.concatenate([
            orthogonal_weight(dim),
            orthogonal_weight(dim),
            orthogonal_weight(dim),
            orthogonal_weight(dim),
            normal_weight(dim, 1)
        ], axis=1)
        for _ in xrange(n_layers - 1)
    ], axis=0)

    params[_p(prefix, 'b')] = np.stack([
        np.zeros(4 * dim + 1, dtype=fX)
        for _ in xrange(n_layers)
    ])

    return params

class _hmrnn_step:
    def __init__(self, n_layers, use_mask=False, use_explicit_boundary=False, trng=None, dropout_rate=0.0):
        self.n_layers = n_layers
        self.use_mask = use_mask
        self.use_explicit_boundary = use_explicit_boundary
        self.trng = trng
        self.dropout_rate = dropout_rate

    def __call__(self, mask, end_mask, x, explicit_boundary,
                 last_h, last_h_dp, last_c, last_z, last_z_before_sigmoid,
                 W0, W_all, U_all, Ut_all, b_all,
                 hard_sigmoid_a, temperature, dim, boundary_type, use_noise):
        '''

            :param mask: batch_size or 1
            :param end_mask: batch_size or 1
            :param x: batch_size * n_in
            :param last_c: n_layers * batch_size * dim
            :param last_z: n_layers * batch_size * 1
            :param W0: dim * (4*dim+1)
            :param W_all: (n_layers-1) * dim * (4*dim+1)
            :param U_all: n_layers * dim * (4*dim+1)`
            :param Ut_all: n_layers * dim * (4*dim+1)
            :param b_all: n_layers * (4*dim+1)
            :param hard_sigmoid_a: scalar
            :param dim: scalar
            :return: h, c, z

        '''

        h_all = []
        h_dp_all = []
        c_all = []
        z_all = []
        z_before_sigmoid_all = []

        for l in xrange(self.n_layers):
            h_l_tm1, h_dp_l_tm1, c_l_tm1, z_l_tm1 = last_h[l], last_h_dp[l], last_c[l], last_z[l]
            if l != 0:
                h_lm1_t, z_lm1_t = h_all[l - 1], z_all[l - 1]
                h_dp_lm1_t = h_dp_all[l - 1]
                W = W_all[l - 1]
            else:
                h_lm1_t, z_lm1_t = x, 1.0
                h_dp_lm1_t = x
                W = W0

            if l != self.n_layers - 1:
                U, Ut, b = U_all[l], Ut_all[l], b_all[l]
                h_lp1_tm1 = last_h[l + 1]
                preact = T.dot(h_l_tm1, U) \
                         + z_l_tm1 * T.dot(h_lp1_tm1, Ut) \
                         + z_lm1_t * T.dot(h_dp_lm1_t, W) \
                         + b
            else:
                U, b = U_all[l], b_all[l]
                preact = T.dot(h_l_tm1, U) \
                         + z_lm1_t * T.dot(h_dp_lm1_t, W) \
                         + b

            f, i, o, g, z = _hmrnn_slice(preact, dim)
            f = T.nnet.sigmoid(f)
            i = T.nnet.sigmoid(i)
            o = T.nnet.sigmoid(o)
            g = T.tanh(g)

            z_before_sigmoid_all.append(z)

            if l == self.n_layers - 1:
                z = T.zeros_like(z, dtype=fX)
                z = T.addbroadcast(z, 1)
            elif self.use_explicit_boundary:
                z = explicit_boundary
                z = T.addbroadcast(z, 1)
            else:
                z_ST = round3(hard_sigmoid(z, hard_sigmoid_a))
                z_Gumbel_Softmax = bernoulli_gumbel_softmax(z, temperature=temperature, trng=self.trng)
                z_ST_Gumbel = round3(bernoulli_gumbel_softmax(z, temperature=temperature, trng=self.trng))
                z_Sigmoid = T.nnet.sigmoid(z)
                z = ifelse(T.eq(boundary_type, 0), z_ST,
                    ifelse(T.eq(boundary_type, 1), z_Gumbel_Softmax,
                    ifelse(T.eq(boundary_type, 2), z_ST_Gumbel,
                                                   z_Sigmoid)))
                z = end_mask[:, None] + (1 - end_mask[:, None]) * z
                z = T.addbroadcast(z, 1)

            fc = f * c_l_tm1
            ig = i * g

            if self.use_mask:
                flush = z_l_tm1
                update_or_copy = (1.0 - z_l_tm1)
                update = update_or_copy * z_lm1_t
                copy = update_or_copy * (1.0 - z_lm1_t)

                update_part_c = (fc + ig)
                copy_part_c = c_l_tm1
                flush_part_c = ig

                c = update * update_part_c + copy * copy_part_c + flush * flush_part_c

                # h = copy * h_l_tm1 + (1.0 - copy) * o * T.tanh(c)
                update_part_h = o * T.tanh(update_part_c)
                copy_part_h = h_l_tm1
                flush_part_h = o * T.tanh(flush_part_c)

                h = update * update_part_h + copy * copy_part_h + flush * flush_part_h
            else:
                c = T.switch(
                    T.gt(z_l_tm1, 0.5),
                    ig,  # flush
                    T.switch(
                        T.gt(z_lm1_t, 0.5),
                        fc + ig,
                        c_l_tm1
                    )
                )

                h = T.switch(
                    T.gt((z_l_tm1 + z_lm1_t), 0.5),
                    o * T.tanh(c),
                    # h_lm1_t
                    h_l_tm1
                )

            if self.dropout_rate > 0.0:
                # h = dropout_layer(h, use_noise, self.trng, self.dropout_rate)
                h_dp = dropout_layer(h, use_noise, self.trng, self.dropout_rate)
            else:
                h_dp = h

            c = mask[:, None] * c + (1.0 - mask)[:, None] * c_l_tm1
            h = mask[:, None] * h + (1.0 - mask)[:, None] * h_l_tm1
            h_dp = mask[:, None] * h_dp + (1.0 - mask)[:, None] * h_dp_l_tm1
            z = mask[:, None] * z + (1.0 - mask)[:, None] * z_l_tm1

            h_all.append(h)
            h_dp_all.append(h_dp)
            c_all.append(c)
            z_all.append(z)

        h = T.stack(h_all, axis=0)
        h_dp = T.stack(h_dp_all, axis=0)
        c = T.stack(c_all, axis=0)
        z = T.stack(z_all, axis=0)
        z_before_sigmoid = T.stack(z_before_sigmoid_all, axis=0)

        z = T.addbroadcast(z, 2)
        z_before_sigmoid = T.addbroadcast(z_before_sigmoid, 2)

        return h, h_dp, c, z, z_before_sigmoid



def hmrnn_layer(P, state_below, O, prefix='hmrnn', mask=None, end_mask=None, explicit_boundary=None, **kwargs):
    '''Hierarchical Multiscale Recurrent Neural Networks (LSTM Cell)

    :param P:
    :param state_below:
    :param O:
    :param prefix:
    :param mask:
    :param kwargs:
    :return:
    '''

    dropout_params = kwargs.pop('dropout_params', None)
    one_step = kwargs.pop('one_step', False)
    init_state = kwargs.pop('init_state', None)
    init_memory = kwargs.pop('init_memory', None)
    init_boundary = kwargs.pop('init_boundary', None)
    hard_sigmoid_a = kwargs.pop('hard_sigmoid_a', 5.0)
    temperature = kwargs.pop('temperature', 1.0)
    n_layers = kwargs.pop('n_layers', 2)
    use_mask = kwargs.pop('use_mask', False)
    use_explicit_boundary = kwargs.pop('use_explicit_boundary', False)
    boundary_type = kwargs.pop('boundary_type', 0)
    trng = kwargs.pop('trng', None)

    kw_ret = {}

    if state_below.ndim == 3:
        n_steps = state_below.shape[0]
        n_samples = state_below.shape[1]
    else:
        n_steps = 1
        n_samples = state_below.shape[0]

    if dropout_params is None:
        use_noise = theano.shared(0.0)
        dropout_rate = 0.0
    else:
        use_noise, drop_trng, dropout_rate = dropout_params
        if trng is None:
            trng = drop_trng

    dim = (P[_p(prefix, 'U')].shape[2] - 1) // 4
    # dim = print_all('Dim Output')(dim)
    mask = T.alloc(1., n_steps, 1) if mask is None else mask
    if end_mask is None:
        end_mask = T.alloc(0., n_steps, 1)
        end_mask = T.set_subtensor(end_mask[-1], T.ones_like(end_mask[-1], dtype=fX))
    if explicit_boundary is None:
        assert not use_explicit_boundary
        explicit_boundary = T.alloc(0., n_steps, 1)  # just a place holder

    seqs = [mask, end_mask, state_below, explicit_boundary]

    init_states = [
        T.alloc(0., n_layers, n_samples, dim) if init_state is None else init_state,
        T.alloc(0., n_layers, n_samples, dim) if init_state is None else init_state,
        T.alloc(0., n_layers, n_samples, dim) if init_memory is None else init_memory,
        T.alloc(0., n_layers, n_samples, 1) if init_boundary is None else init_boundary,
        T.alloc(0., n_layers, n_samples, 1)
    ]

    params = [
        P[_p(prefix, 'W0')],
        P[_p(prefix, 'W')],
        P[_p(prefix, 'U')],
        P[_p(prefix, 'Ut')],
        P[_p(prefix, 'b')],
    ]
    shared_vars = params + [hard_sigmoid_a, temperature, dim, boundary_type, use_noise]
    _step = _hmrnn_step(
        n_layers=n_layers,
        use_mask=use_mask,
        use_explicit_boundary=use_explicit_boundary,
        trng=trng,
        dropout_rate=dropout_rate
    )
    if one_step:
        outputs = _step(*(seqs + init_states + shared_vars))
        kw_ret['all_layer_h'] = outputs[0]
        kw_ret['all_layer_h_dp'] = outputs[1]
        kw_ret['all_layer_c'] = outputs[2]
        kw_ret['all_layer_z'] = outputs[3]
        kw_ret['all_layer_z_before_sigmoid'] = outputs[4]
        kw_ret['updates'] = OrderedUpdates()
        highest_layer = outputs[1][-1]
        second_highest_boundary = outputs[3][-2]
    else:
        outputs, updates = theano.scan(
            _step,
            sequences=seqs,
            outputs_info=init_states,
            non_sequences=shared_vars,
            name=_p(prefix, 'layers'),
            n_steps=n_steps,
            profile=profile,
            strict=True,
        )
        kw_ret['all_layer_h'] = outputs[0]
        kw_ret['all_layer_h_dp'] = outputs[1]
        kw_ret['all_layer_c'] = outputs[2]
        kw_ret['all_layer_z'] = outputs[3]
        kw_ret['all_layer_z_before_sigmoid'] = outputs[4]
        kw_ret['updates'] = updates
        highest_layer = outputs[1][:, -1]
        second_highest_boundary = outputs[3][:, -2]

    # if dropout_params:
    #     highest_layer = dropout_layer(highest_layer, *dropout_params)
    return highest_layer, second_highest_boundary, kw_ret

def param_init_hmrnn_cond(O, params, prefix='hmrnn_cond', nin=None, dim=None, dimctx=None, **kwargs):
    if nin is None:
        nin = O['dim']
    if dim is None:
        dim = O['dim']
    if dimctx is None:
        dimctx = O['dim']
    n_layers = kwargs.pop('n_layers', 2)

    params[_p(prefix, 'W0')] = np.concatenate([
        normal_weight(nin, dim),
        normal_weight(nin, dim),
        normal_weight(nin, dim),
        normal_weight(nin, dim),
        normal_weight(nin, 1)
    ], axis=1)

    params[_p(prefix, 'W')] = np.stack([
        np.concatenate([
            normal_weight(dim, dim),
            normal_weight(dim, dim),
            normal_weight(dim, dim),
            normal_weight(dim, dim),
            normal_weight(dim, 1)
        ], axis=1)
        for _ in xrange(n_layers - 1)
    ], axis=0)

    params[_p(prefix, 'U')] = np.stack([
        np.concatenate([
            orthogonal_weight(dim),
            orthogonal_weight(dim),
            orthogonal_weight(dim),
            orthogonal_weight(dim),
            normal_weight(dim, 1)
        ], axis=1)
        for _ in xrange(n_layers)
    ], axis=0)

    params[_p(prefix, 'Ut')] = np.stack([
        np.concatenate([
            orthogonal_weight(dim),
            orthogonal_weight(dim),
            orthogonal_weight(dim),
            orthogonal_weight(dim),
            normal_weight(dim, 1)
        ], axis=1)
        for _ in xrange(n_layers - 1)
    ], axis=0)

    # context to LSTM
    params[_p(prefix, 'Wc')] = np.stack([
        np.concatenate([
            normal_weight(dimctx, dim),
            normal_weight(dimctx, dim),
            normal_weight(dimctx, dim),
            normal_weight(dimctx, dim),
            normal_weight(dimctx, 1)
        ], axis=1)
        for _ in xrange(n_layers)
    ], axis=0)

    params[_p(prefix, 'b')] = np.stack([
        np.zeros(4 * dim + 1, dtype=fX)
        for _ in xrange(n_layers)
    ])

    # attention: combined -> hidden
    if not O['layerwise_attention']:
        if O['decoder_all_attention']:
            params[_p(prefix, 'W_comb_att')] = normal_weight(n_layers * dim, dimctx)
        else:
            params[_p(prefix, 'W_comb_att')] = normal_weight(dim, dimctx)

        # attention: context -> hidden
        params[_p(prefix, 'Wc_att')] = normal_weight(dimctx)

        # attention: hidden bias
        params[_p(prefix, 'b_att')] = np.zeros((dimctx,), dtype=fX)

        # attention:
        params[_p(prefix, 'U_att')] = normal_weight(dimctx, 1)
        params[_p(prefix, 'c_tt')] = np.zeros((1,), dtype=fX)
    else:
        params[_p(prefix, 'W_comb_att')] = np.stack([
            normal_weight(dim, dimctx)
            for _ in xrange(n_layers)
        ])

        # attention: context -> hidden
        params[_p(prefix, 'Wc_att')] = np.stack([
            normal_weight(dimctx)
            for _ in xrange(n_layers)
        ])

        # attention: hidden bias
        params[_p(prefix, 'b_att')] = np.stack([
            np.zeros((dimctx,), dtype=fX)
            for _ in xrange(n_layers)
        ])

        # attention:
        params[_p(prefix, 'U_att')] = np.stack([
            normal_weight(dimctx, 1)
            for _ in xrange(n_layers)
        ])
        params[_p(prefix, 'c_tt')] = np.stack([
            np.zeros((1,), dtype=fX)
            for _ in xrange(n_layers)
        ])

    return params

def _attention(h1, projected_context_, context_, W_comb_att, U_att, c_tt, context_mask=None):
    pstate_ = T.dot(h1, W_comb_att)
    pctx__ = projected_context_ + pstate_[None, :, :]
    pctx__ = T.tanh(pctx__)

    alpha = T.dot(pctx__, U_att) + c_tt

    alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
    alpha = T.exp(alpha - alpha.max(axis=0, keepdims=True))
    if context_mask:
        # context_mask = theano.gradient.disconnected_grad(context_mask)
        # context_mask = T.ones_like(context_mask, dtype=fX)
        alpha = alpha * context_mask
    alpha = alpha / alpha.sum(0, keepdims=True)
    ctx_ = (context_ * alpha[:, :, None]).sum(0)  # current context

    return ctx_, alpha.T

class _hmrnn_cond_step:
    def __init__(self, n_layers, use_mask=False, use_explicit_boundary=False, all_att=False,
                 use_all_one_boundary=False, layerwise_attention=False, trng=None, dropout_rate=0.0):
        self.n_layers = n_layers
        self.use_mask = use_mask
        self.use_explicit_boundary = use_explicit_boundary
        self.all_att = all_att
        self.use_all_one_boundary = use_all_one_boundary
        self.layerwise_attention = layerwise_attention
        self.trng = trng
        self.dropout_rate = dropout_rate

    def __call__(self, mask, x, explicit_boundary,
                 last_h, last_h_dp, last_c, last_z, last_ctx, last_alpha, last_z_before_sigmoid,
                 W0, W_all, U_all, Ut_all, b_all, Wc_all,
                 W_comb_att, U_att, c_tt,
                 projected_context, context, context_mask,
                 hard_sigmoid_a, temperature, dim, boundary_type, use_noise):
        '''

            :param mask: batch_size or 1
            :param x: batch_size * n_in
            :param last_h: n_layers * batch_size * dim
            :param last_c: n_layers * batch_size * dim
            :param last_z: n_layers * batch_size * 1
            :param W0: dim * (4*dim+1)
            :param W_all: (n_layers-1) * dim * (4*dim+1)
            :param U_all: n_layers * dim * (4*dim+1)
            :param Ut_all: n_layers * dim * (4*dim+1)
            :param b_all: n_layers * (4*dim+1)
            :param hard_sigmoid_a: scalar
            :param dim: scalar
            :return: h, c, z

        '''
        h_all = []
        h_dp_all = []
        c_all = []
        z_all = []
        z_before_sigmoid_all = []

        if self.layerwise_attention:
            ctx_all = []
            alpha_all = []
        elif self.all_att:
            attention_hidden = last_h.dimshuffle(1, 0, 2).reshape((last_h.shape[1], -1))
            ctx, alpha = _attention(attention_hidden, projected_context, context,
                                    W_comb_att, U_att, c_tt, context_mask=context_mask)
        else:
            ctx, alpha = _attention(last_h[-1], projected_context, context,
                                    W_comb_att, U_att, c_tt, context_mask=context_mask)

        for l in xrange(self.n_layers):
            if self.layerwise_attention:
                ctx, alpha = _attention(last_h[l], projected_context[l], context[l],
                                        W_comb_att[l], U_att[l], c_tt[l], context_mask=context_mask[l])
                ctx_all.append(ctx)
                alpha_all.append(alpha)

            h_l_tm1, h_dp_l_tm1, c_l_tm1, z_l_tm1 = last_h[l], last_h_dp[l], last_c[l], last_z[l]
            if l != 0:
                h_lm1_t, z_lm1_t = h_all[l - 1], z_all[l - 1]
                h_dp_lm1_t = h_dp_all[l - 1]
                W = W_all[l - 1]
            else:
                h_lm1_t, z_lm1_t = x, 1.0
                h_dp_lm1_t = x
                W = W0

            if l != self.n_layers - 1:
                U, Wc, Ut, b = U_all[l], Wc_all[l], Ut_all[l], b_all[l]
                h_lp1_tm1 = last_h[l + 1]
                preact = T.dot(h_l_tm1, U) \
                         + T.dot(ctx, Wc) \
                         + z_l_tm1 * T.dot(h_lp1_tm1, Ut) \
                         + z_lm1_t * T.dot(h_dp_lm1_t, W) \
                         + b
            else:
                U, Wc, b = U_all[l], Wc_all[l], b_all[l]
                preact = T.dot(h_l_tm1, U) \
                         + T.dot(ctx, Wc) \
                         + z_lm1_t * T.dot(h_dp_lm1_t, W) \
                         + b
            f, i, o, g, z = _hmrnn_slice(preact, dim)
            f = T.nnet.sigmoid(f)
            i = T.nnet.sigmoid(i)
            o = T.nnet.sigmoid(o)
            g = T.tanh(g)

            z_before_sigmoid_all.append(z)

            if l == self.n_layers - 1:
                z = T.zeros_like(z, dtype=fX)
                z = T.addbroadcast(z, 1)
            elif self.use_explicit_boundary:
                z = explicit_boundary
                z = T.addbroadcast(z, 1)
            elif self.use_all_one_boundary:
                z = T.ones_like(z, dtype=fX)
                z = T.addbroadcast(z, 1)
            else:
                z_ST = round3(hard_sigmoid(z, hard_sigmoid_a))
                z_Gumbel_Softmax = bernoulli_gumbel_softmax(z, temperature=temperature, trng=self.trng)
                z_ST_Gumbel = round3(bernoulli_gumbel_softmax(z, temperature=temperature, trng=self.trng))
                z_Sigmoid = T.nnet.sigmoid(z)
                z = ifelse(T.eq(boundary_type, 0), z_ST,
                    ifelse(T.eq(boundary_type, 1), z_Gumbel_Softmax,
                    ifelse(T.eq(boundary_type, 2), z_ST_Gumbel,
                                                   z_Sigmoid)))

                # z = theano.printing.Print('z', attrs=['shape', '__str__'])(z)

                z = T.addbroadcast(z, 1)

            fc = f * c_l_tm1
            ig = i * g

            if self.use_mask:
                flush = z_l_tm1
                update_or_copy = (1.0 - z_l_tm1)
                update = update_or_copy * z_lm1_t
                copy = update_or_copy * (1.0 - z_lm1_t)

                update_part_c = (fc + ig)
                copy_part_c = c_l_tm1
                flush_part_c = ig

                c = update * update_part_c + copy * copy_part_c + flush * flush_part_c

                # h = copy * h_l_tm1 + (1.0 - copy) * o * T.tanh(c)
                update_part_h = o * T.tanh(update_part_c)
                copy_part_h = h_l_tm1
                flush_part_h = o * T.tanh(flush_part_c)

                h = update * update_part_h + copy * copy_part_h + flush * flush_part_h
            else:
                c = T.switch(
                    T.gt(z_l_tm1, 0.5),
                    ig,  # flush
                    T.switch(
                        T.gt(z_lm1_t, 0.5),
                        fc + ig,
                        c_l_tm1
                    )
                )

                h = T.switch(
                    T.gt((z_l_tm1 + z_lm1_t), 0.5),
                    o * T.tanh(c),
                    # h_lm1_t
                    h_l_tm1
                )

            if self.dropout_rate > 0.0:
                # h = dropout_layer(h, use_noise, self.trng, self.dropout_rate)
                h_dp = dropout_layer(h, use_noise, self.trng, self.dropout_rate)
            else:
                h_dp = h

            c = mask[:, None] * c + (1.0 - mask)[:, None] * c_l_tm1
            h = mask[:, None] * h + (1.0 - mask)[:, None] * h_l_tm1
            h_dp = mask[:, None] * h_dp + (1.0 - mask)[:, None] * h_dp_l_tm1
            z = mask[:, None] * z + (1.0 - mask)[:, None] * z_l_tm1

            h_all.append(h)
            h_dp_all.append(h_dp)
            c_all.append(c)
            z_all.append(z)

        h = T.stack(h_all, axis=0)
        h_dp = T.stack(h_dp_all, axis=0)
        c = T.stack(c_all, axis=0)
        z = T.stack(z_all, axis=0)
        z_before_sigmoid = T.stack(z_before_sigmoid_all, axis=0)

        z = T.addbroadcast(z, 2)
        z_before_sigmoid = T.addbroadcast(z_before_sigmoid, 2)
        if self.layerwise_attention:
            ctx = concatenate(ctx_all, axis=1)
            alpha = T.stack(alpha_all, axis=0).mean(axis=0)
        return h, h_dp, c, z, ctx, alpha, z_before_sigmoid

def hmrnn_cond_layer(P, state_below, O, prefix='hmrnn_cond', mask=None, context=None, one_step=False,
                     init_memory=None, init_state=None, init_boundary=None,
                     context_mask=None, explicit_boundary=None, **kwargs):
    kw_ret = {}
    hard_sigmoid_a = kwargs.pop('hard_sigmoid_a', 5.0)
    temperature = kwargs.pop('temperature', 1.0)
    n_layers = kwargs.pop('n_layers', 2)
    dropout_params = kwargs.pop('dropout_params', None)
    use_mask = kwargs.pop('use_mask', False)
    use_explicit_boundary = kwargs.pop('use_explicit_boundary', False)
    boundary_type = kwargs.pop('boundary_type', 0)
    trng = kwargs.pop('trng', None)
    benefit_0_boundary = kwargs.pop('benefit_0_boundary', False)
    all_att = kwargs.pop('all_att', False)
    use_all_one_boundary = kwargs.pop('use_all_one_boundary', False)
    layerwise_attention = kwargs.pop('layerwise_attention', False)

    assert context, 'Context must be provided'
    if not layerwise_attention:
        assert context.ndim == 3, 'Context must be 3-d: #annotation * #sample * dim'
    else:
        assert context.ndim == 4, 'Context must be 4-d: #layer * #annotation * #sample * dim'
    if one_step:
        assert init_state, 'previous state must be provided'

    if state_below.ndim == 3:
        n_steps = state_below.shape[0]
        n_samples = state_below.shape[1]
    else:
        n_steps = 1
        n_samples = state_below.shape[0]

    if dropout_params is None:
        use_noise = theano.shared(0.0)
        dropout_rate = 0.0
    else:
        use_noise, drop_trng, dropout_rate = dropout_params
        if trng is None:
            trng = drop_trng

    dim = (P[_p(prefix, 'U')].shape[2] - 1) // 4

    if mask is None:
        mask = T.alloc(1., n_steps, 1)

    if explicit_boundary is None:
        assert not use_explicit_boundary
        explicit_boundary = T.alloc(0., n_steps, 1)  # just a place holder

    if not layerwise_attention:
        projected_context = T.dot(context, P[_p(prefix, 'Wc_att')]) + P[_p(prefix, 'b_att')]
    else:
        projected_contexts = []
        for l in xrange(n_layers):
            projected_contexts.append(T.dot(context[l], P[_p(prefix, 'Wc_att')][l]) + P[_p(prefix, 'b_att')][l])
        projected_context = T.stack(projected_contexts, axis=0)

    if init_state is None:
        init_state = T.alloc(0., n_layers, n_samples, dim)
    if init_memory is None:
        init_memory = T.alloc(0., n_layers, n_samples, dim)
    if init_boundary is None:
        init_boundary = T.alloc(0., n_layers, n_samples, 1)

    seqs = [mask, state_below, explicit_boundary]

    init_states = [
        init_state,
        init_state,
        init_memory,
        init_boundary,
        T.alloc(0., n_samples, context.shape[2])
            if not layerwise_attention else T.alloc(0., n_samples, n_layers * context.shape[3]),
        T.alloc(0., n_samples, context.shape[0])
            if not layerwise_attention else T.alloc(0., n_samples, context.shape[1]),
        T.alloc(0., n_layers, n_samples, 1)
    ]

    params = [
        P[_p(prefix, 'W0')],
        P[_p(prefix, 'W')],
        P[_p(prefix, 'U')],
        P[_p(prefix, 'Ut')],
        P[_p(prefix, 'b')],
        P[_p(prefix, 'Wc')],
        P[_p(prefix, 'W_comb_att')],
        P[_p(prefix, 'U_att')],
        P[_p(prefix, 'c_tt')],
    ]
    if context_mask is None:
        context_mask = T.alloc(1., context.shape[0], context.shape[1])
    shared_vars = params + [projected_context, context, context_mask,
                            hard_sigmoid_a, temperature, dim, boundary_type, use_noise]
    _step = _hmrnn_cond_step(n_layers=n_layers, use_mask=use_mask, all_att=all_att,
                             use_explicit_boundary=use_explicit_boundary, use_all_one_boundary=use_all_one_boundary,
                             trng=trng, dropout_rate=dropout_rate, layerwise_attention=layerwise_attention)
    if one_step:
        result = _step(*(seqs + init_states + shared_vars))
        kw_ret['all_layer_h'] = result[0]
        kw_ret['all_layer_h_dp'] = result[1]
        kw_ret['all_layer_c'] = result[2]
        kw_ret['all_layer_z'] = result[3]
        kw_ret['all_layer_z_before_sigmoid'] = result[6]
        kw_ret['updates'] = OrderedUpdates()
        h_ret = result[1]
        if benefit_0_boundary:
            print 'benefit_0_boundary'
            h_ret = (1.0 - result[3]) * h_ret
        h_ret = h_ret.dimshuffle(1, 0, 2)
        h_ret = h_ret.reshape((result[1].shape[1], -1))

    else:
        result, updates = theano.scan(
            _step,
            sequences=seqs,
            outputs_info=init_states,
            non_sequences=shared_vars,
            name=_p(prefix, 'layers'),
            n_steps=n_steps,
            profile=profile,
            strict=True,
        )
        kw_ret['all_layer_h'] = result[0]
        kw_ret['all_layer_h_dp'] = result[1]
        kw_ret['all_layer_c'] = result[2]
        kw_ret['all_layer_z'] = result[3]
        kw_ret['all_layer_z_before_sigmoid'] = result[6]
        kw_ret['updates'] = updates
        h_ret = result[1]
        if benefit_0_boundary:
            print 'benefit_0_boundary'
            h_ret = (1.0 - result[3]) * h_ret
        h_ret = h_ret.dimshuffle(0, 2, 1, 3)
        h_ret = h_ret.reshape((result[1].shape[0], result[1].shape[2], -1))


    # if dropout_params:
    #     h_ret = dropout_layer(h_ret, *dropout_params)

    return h_ret, result[4], result[5], kw_ret

