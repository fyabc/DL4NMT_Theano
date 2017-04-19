#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

from collections import OrderedDict
import sys
import os
import copy

import theano
import theano.tensor as T
import numpy as np
import cPickle as pkl
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from constants import fX, profile
from utils import *
from layers import *

__author__ = 'fyabc'


class ParameterInitializer(object):
    def __init__(self, options):
        # Dict of options
        self.O = options

    @staticmethod
    def init_embedding(np_parameters, name, n_in, n_out):
        np_parameters[name] = normal_weight(n_in, n_out)

    def init_encoder(self, np_parameters):
        if self.O['encoder_many_bidirectional']:
            for layer_id in xrange(self.O['n_encoder_layers']):
                if layer_id == 0:
                    n_in = self.O['dim_word']
                else:
                    n_in = self.O['dim']
                np_parameters = get_init(self.O['unit'])(self.O, np_parameters, prefix='encoder', nin=n_in,
                                                         dim=self.O['dim'], layer_id=layer_id)
                np_parameters = get_init(self.O['unit'])(self.O, np_parameters, prefix='encoder_r', nin=n_in,
                                                         dim=self.O['dim'], layer_id=layer_id)
        else:
            for layer_id in xrange(self.O['n_encoder_layers']):
                if layer_id == 0:
                    n_in = self.O['dim_word']
                    np_parameters = get_init(self.O['unit'])(self.O, np_parameters, prefix='encoder', nin=n_in,
                                                             dim=self.O['dim'], layer_id=0)
                    np_parameters = get_init(self.O['unit'])(self.O, np_parameters, prefix='encoder_r', nin=n_in,
                                                             dim=self.O['dim'], layer_id=0)
                else:
                    n_in = 2 * self.O['dim']
                    np_parameters = get_init(self.O['unit'])(self.O, np_parameters, prefix='encoder', nin=n_in,
                                                             dim=n_in, layer_id=layer_id)

        return np_parameters

    def init_decoder(self, np_parameters):
        context_dim = 2 * self.O['dim']
        attention_layer_id = self.O['attention_layer_id']

        # init_state, init_cell
        np_parameters = self.init_feed_forward(np_parameters, prefix='ff_state', nin=context_dim, nout=self.O['dim'])

        # Layers before attention layer
        for layer_id in xrange(0, attention_layer_id):
            np_parameters = get_init(self.O['unit'])(
                self.O, np_parameters, prefix='decoder', nin=self.O['dim_word'] if layer_id == 0 else self.O['dim'],
                dim=self.O['dim'], layer_id=layer_id, context_dim=None)

        # Attention layer
        np_parameters = get_init(self.O['unit'] + '_cond')(
            self.O, np_parameters, prefix='decoder',
            nin=self.O['dim_word'] if attention_layer_id == 0 else self.O['dim'],
            dim=self.O['dim'], dimctx=context_dim, layer_id=attention_layer_id)

        # Layers after attention layer
        for layer_id in xrange(attention_layer_id + 1, self.O['n_decoder_layers']):
            np_parameters = get_init(self.O['unit'])(
                self.O, np_parameters, prefix='decoder', nin=self.O['dim'],
                dim=self.O['dim'], layer_id=layer_id, context_dim=context_dim)

        return np_parameters

    def init_input_to_context(self, parameters, reload_=None, preload=None,
                              load_embedding=True):
        """Initialize the model parameters from input to context vector.

        :param parameters: OrderedDict of Theano shared variables to be initialized.
        :param reload_: Reload old model or not
        :param preload: The path of old model
        :param load_embedding: Load old word embedding or not, default to True
        """

        np_parameters = OrderedDict()

        # Source embedding
        self.init_embedding(np_parameters, 'Wemb', self.O['n_words_src'], self.O['dim_word'])

        # Encoder: bidirectional RNN
        np_parameters = self.init_encoder(np_parameters)

        # Reload parameters
        reload_ = self.O['reload_'] if reload_ is None else reload_
        preload = self.O['preload'] if preload is None else preload
        if reload_ and os.path.exists(preload):
            print('Reloading model parameters')
            np_parameters = load_params(preload, np_parameters)
        else:
            if load_embedding:
                # [NOTE] Important: Load embedding even in random init case
                print('Loading embedding')
                old_params = np.load(self.O['preload'])
                np_parameters['Wemb'] = old_params['Wemb']

        print_params(np_parameters)

        # Init theano parameters
        init_tparams(np_parameters, parameters)

    def init_input_to_decoder_context(self, parameters, reload_=None, preload=None, load_embedding=True):
        """Initialize the model parameters from input to decoder context vector.

        :param parameters: OrderedDict of Theano shared variables to be initialized.
        :param reload_: Reload old model or not
        :param preload: The path of old model
        :param load_embedding: Load old word embedding or not, default to True
        """

        np_parameters = OrderedDict()

        # Source embedding
        self.init_embedding(np_parameters, 'Wemb', self.O['n_words_src'], self.O['dim_word'])

        # Encoder: bidirectional RNN
        np_parameters = self.init_encoder(np_parameters)

        # Target embedding
        self.init_embedding(np_parameters, 'Wemb_dec', self.O['n_words'], self.O['dim_word'])

        # Decoder
        np_parameters = self.init_decoder(np_parameters)

        # Reload parameters
        reload_ = self.O['reload_'] if reload_ is None else reload_
        preload = self.O['preload'] if preload is None else preload
        if reload_ and os.path.exists(preload):
            print('Reloading model parameters')
            np_parameters = load_params(preload, np_parameters)
        else:
            if load_embedding:
                # [NOTE] Important: Load embedding even in random init case
                print('Loading embedding')
                old_params = np.load(self.O['preload'])
                np_parameters['Wemb'] = old_params['Wemb']
                np_parameters['Wemb_dec'] = old_params['Wemb_dec']

        print_params(np_parameters)

        # Init theano parameters
        init_tparams(np_parameters, parameters)

    def init_params(self):
        np_parameters = OrderedDict()

        # Source embedding
        self.init_embedding(np_parameters, 'Wemb', self.O['n_words_src'], self.O['dim_word'])

        # Encoder: bidirectional RNN
        np_parameters = self.init_encoder(np_parameters)

        # Target embedding
        self.init_embedding(np_parameters, 'Wemb_dec', self.O['n_words'], self.O['dim_word'])

        # Decoder
        np_parameters = self.init_decoder(np_parameters)

        # Readout
        context_dim = 2 * self.O['dim']
        np_parameters = self.init_feed_forward(np_parameters, prefix='ff_logit_lstm', nin=self.O['dim'],
                                               nout=self.O['dim_word'], orthogonal=False)
        np_parameters = self.init_feed_forward(np_parameters, prefix='ff_logit_prev', nin=self.O['dim_word'],
                                               nout=self.O['dim_word'], orthogonal=False)
        np_parameters = self.init_feed_forward(np_parameters, prefix='ff_logit_ctx', nin=context_dim,
                                               nout=self.O['dim_word'], orthogonal=False)
        np_parameters = self.init_feed_forward(np_parameters, prefix='ff_logit', nin=self.O['dim_word'],
                                               nout=self.O['n_words'])

        return np_parameters

    def init_feed_forward(self, params, prefix='ff', nin=None, nout=None, orthogonal=True):
        """feed-forward layer: affine transformation + point-wise nonlinearity"""

        if nin is None:
            nin = self.O['dim_proj']
        if nout is None:
            nout = self.O['dim_proj']
        params[_p(prefix, 'W')] = normal_weight(nin, nout, scale=0.01, orthogonal=orthogonal)
        params[_p(prefix, 'b')] = np.zeros((nout,), dtype=fX)

        return params


class NMTModel(object):
    """The model class.

    This is a light weight class, just contains some needed elements and model components.
    The main work is done by the caller.
    
    Shapes of some inputs, outputs and intermediate results:
        [W]: dim_word, word vector dim, 100
        [H]: dim, hidden size, 1000
        [BS]: batch_size or n_samples, 80
        [Ts]: n_timestep, 100
        [Tt]: n_timestep_tgt, 100
        [Hc]: context_dim, 2 * [H]
        
        x, x_mask: ([Ts], [BS])
        y, y_mask: ([Tt], [BS])
        
        src_embedding: ([Ts], [BS], [W])
        
        # Encoder
            # todo
        
        context: ([Ts], [BS], [Hc])
        
        context_mean: ([BS], [Hc])
        init_state: ([BS], [H])
        
        tgt_embedding: ([Tt], [BS], [W])
        
        # Decoder
            # Attention: context -> hidden
            Wc_att: ([Hc], [Hc])
            b_att: ([Hc])
            projected_context: ([Tt], [BS], [Hc])
            
            # GRU1 (without attention)
            # [H] + [H]: combine reset and update gates
            U: ([H], [H] + [H])
            b: ([H] + [H])
            ProjectedTargetEmbedding: ([Tt], [BS], [H] + [H]) -> alias to x
            x_t: ([BS], [H] + [H])
            
            Ux: ([H], [H])
            bx: ([H])
            ProjectedTargetEmbedding_proj: ([Tt], [BS], [H]) -> alias to x_proj
            x_proj_t: ([BS], [H])
            
            r1_t, u1_t, h1_t: ([BS], [H])
            
            # Attention combined -> hidden
            W_comb_att: ([H], [Hc])
            pstate_: ([BS], [Hc])
            pctx__: ([Tt], [BS], [Hc])
            
            U_att: ([Hc], 1)
            c_tt: (1)
            alpha: ([Tt], [BS])
            
            # Current context
            ctx_: ([BS], [Hc])
            
            # GRU2 (with attention)
            U_nl: ([H], [H] + [H])
            b_nl: ([H] + [H])
            Wc: ([Hc], [H] + [H])
            
            Ux_nl: ([H], [H])
            bx_nl: ([H])
            Wcx: ([Hc], [H])
            
            r2_t, u2_t, h2_t: ([BS], [H])
            
            h_t: ([BS], [H])
            
        hidden_decoder: ([Tt], [BS], [H])
        context_decoder: ([Tt], [BS], [Hc])
        alpha_decoder: ([Tt], [Bs], [Tt])
    """

    # This is a simple wrapper of layers.py now.
    # todo: Move code from layers.py to here.

    def __init__(self, options, given_params=None):
        # Dict of options
        self.O = options

        # Dict of parameters (Theano shared variables)
        self.P = OrderedDict() if given_params is None else given_params

        # Instance of ParameterInitializer, init the parameters.
        self.initializer = ParameterInitializer(options)

    def input_to_context(self, given_input=None, **kwargs):
        """Build the part of the model that from input to context vector.

        Used for regression of deeper encoder.

        :param given_input: List of input Theano tensors or None
            If None, this method will create them by itself.
        :returns tuple of input list and output
        """

        x, x_mask, y, y_mask = self.get_input() if given_input is None else given_input

        # For the backward rnn, we just need to invert x and x_mask
        x_r, x_mask_r = self.reverse_input(x, x_mask)

        n_timestep, n_timestep_tgt, n_samples = self.input_dimensions(x, y)

        # Word embedding for forward rnn and backward rnn (source)
        src_embedding = self.embedding(x, n_timestep, n_samples)
        src_embedding_r = self.embedding(x_r, n_timestep, n_samples)

        # Encoder
        context = self.encoder(src_embedding, src_embedding_r, x_mask, x_mask_r,
                               dropout_params=kwargs.pop('dropout_params', None))

        return [x, x_mask, y, y_mask], context

    def input_to_decoder_context(self, given_input=None):
        """Build the part of the model that from input to context vector of decoder.

        :param given_input: List of input Theano tensors or None
            If None, this method will create them by itself.
        :return: tuple of input list and output
        """

        (x, x_mask, y, y_mask), context = self.input_to_context(given_input)

        n_timestep, n_timestep_tgt, n_samples = self.input_dimensions(x, y)

        # Mean of the context (across time) will be used to initialize decoder rnn
        context_mean = self.get_context_mean(context, x_mask)
        # Initial decoder state
        init_decoder_state = self.feed_forward(context_mean, prefix='ff_state', activation=tanh)

        # Word embedding (target), we will shift the target sequence one time step
        # to the right. This is done because of the bi-gram connections in the
        # readout and decoder rnn. The first target will be all zeros and we will
        # not condition on the last output.
        tgt_embedding = self.embedding(y, n_timestep_tgt, n_samples, 'Wemb_dec')
        emb_shifted = T.zeros_like(tgt_embedding)
        emb_shifted = T.set_subtensor(emb_shifted[1:], tgt_embedding[:-1])
        tgt_embedding = emb_shifted

        # Decoder - pass through the decoder conditional gru with attention
        hidden_decoder, context_decoder, _, _ = self.decoder(
            tgt_embedding, y_mask, init_decoder_state, context, x_mask,
            dropout_params=None,
        )

        return [x, x_mask, y, y_mask], hidden_decoder, context_decoder

    def init_tparams(self, np_parameters):
        self.P = init_tparams(np_parameters)

    def build_model(self):
        """Build a training model."""

        dropout_rate = self.O['use_dropout']

        opt_ret = {}

        trng = RandomStreams(1234)
        use_noise = theano.shared(np.float32(0.))

        if dropout_rate is not False:
            dropout_params = [use_noise, trng, dropout_rate]
        else:
            dropout_params = None

        (x, x_mask, y, y_mask), context = self.input_to_context(dropout_params=dropout_params)

        n_timestep, n_timestep_tgt, n_samples = self.input_dimensions(x, y)

        context_mean = self.get_context_mean(context, x_mask)
        # Initial decoder state
        init_decoder_state = self.feed_forward(context_mean, prefix='ff_state', activation=tanh)

        # Word embedding (target), we will shift the target sequence one time step
        # to the right. This is done because of the bi-gram connections in the
        # readout and decoder rnn. The first target will be all zeros and we will
        # not condition on the last output.
        tgt_embedding = self.embedding(y, n_timestep_tgt, n_samples, 'Wemb_dec')
        emb_shifted = T.zeros_like(tgt_embedding)
        emb_shifted = T.set_subtensor(emb_shifted[1:], tgt_embedding[:-1])
        tgt_embedding = emb_shifted

        # Decoder - pass through the decoder conditional gru with attention
        hidden_decoder, context_decoder, opt_ret['dec_alphas'], _ = self.decoder(
            tgt_embedding, y_mask=y_mask, init_state=init_decoder_state, context=context, x_mask=x_mask,
            dropout_params=dropout_params, one_step=False,
        )

        trng, use_noise, probs = self.get_word_probability(hidden_decoder, context_decoder, tgt_embedding,
                                                           trng=trng, use_noise=use_noise)

        cost = self.build_cost(y, y_mask, probs)

        # Plot computation graph
        if self.O['plot_graph'] is not None:
            print('Plotting pre-compile graph...', end='')
            theano.printing.pydotprint(
                cost,
                outfile='pictures/pre_compile_{}'.format(self.O['plot_graph']),
                var_with_name_simple=True,
            )
            print('Done')

        return trng, use_noise, x, x_mask, y, y_mask, opt_ret, cost, context_mean

    def build_sampler(self, **kwargs):
        """Build a sampler."""

        unit = self.O['unit']

        trng = kwargs.pop('trng', RandomStreams(1234))
        use_noise = kwargs.pop('use_noise', theano.shared(np.float32(0.)))

        x = T.matrix('x', dtype='int64')
        xr = x[::-1]
        n_timestep = x.shape[0]
        n_samples = x.shape[1]

        # Word embedding for forward rnn and backward rnn (source)
        src_embedding = self.embedding(x, n_timestep, n_samples)
        src_embedding_r = self.embedding(xr, n_timestep, n_samples)

        # Encoder
        ctx = self.encoder(src_embedding, src_embedding_r, None, None, dropout_params=None)

        # Get the input for decoder rnn initializer mlp
        ctx_mean = ctx.mean(0)
        init_state = self.feed_forward(ctx_mean, prefix='ff_state', activation=tanh)

        print('Building f_init...', end='')
        outs = [init_state, ctx]
        f_init = theano.function([x], outs, name='f_init', profile=profile)
        print('Done')

        # x: 1 x 1
        y = T.vector('y_sampler', dtype='int64')
        init_state = T.tensor3('init_state', dtype=fX)
        init_memory = T.tensor3('init_memory', dtype=fX)

        # If it's the first word, emb should be all zero and it is indicated by -1
        emb = T.switch(y[:, None] < 0,
                       T.alloc(0., 1, self.P['Wemb_dec'].shape[1]),
                       self.P['Wemb_dec'][y])

        # Apply one step of conditional gru with attention
        hidden_decoder, context_decoder, _, kw_ret = self.decoder(
            emb, y_mask=None, init_state=init_state, context=ctx, x_mask=None,
            dropout_params=None, one_step=True, init_memory=init_memory,
        )

        # Get memory_out and hiddens_without_dropout
        # FIXME: stack list into a single tensor
        memory_out = None
        hiddens_without_dropout = T.stack(kw_ret['hiddens_without_dropout'])
        if unit == 'lstm':
            memory_out = T.stack(kw_ret['memory_outputs'])

        logit_lstm = self.feed_forward(hidden_decoder, prefix='ff_logit_lstm', activation=linear)
        logit_prev = self.feed_forward(emb, prefix='ff_logit_prev', activation=linear)
        logit_ctx = self.feed_forward(context_decoder, prefix='ff_logit_ctx', activation=linear)
        logit = T.tanh(logit_lstm + logit_prev + logit_ctx)
        if self.O['use_dropout']:
            logit = self.dropout(logit, use_noise, trng)
        logit = self.feed_forward(logit, prefix='ff_logit', activation=linear)

        # Compute the softmax probability
        next_probs = T.nnet.softmax(logit)

        # Sample from softmax distribution to get the sample
        next_sample = trng.multinomial(pvals=next_probs).argmax(1)

        # Compile a function to do the whole thing above, next word probability,
        # sampled word for the next target, next hidden state to be used
        print('Building f_next..', end='')
        inps = [y, ctx, init_state]
        outs = [next_probs, next_sample, hiddens_without_dropout]
        if unit == 'lstm':
            inps.append(init_memory)
            outs.append(memory_out)
        f_next = theano.function(inps, outs, name='f_next', profile=profile)
        print('Done')

        return f_init, f_next

    def gen_sample(self, f_init, f_next, x, trng=None, k=1, maxlen=30,
                   stochastic=True, argmax=False):
        """Generate sample, either with stochastic sampling or beam search. Note that,

        this function iteratively calls f_init and f_next functions.
        """

        unit = self.O['unit']

        # k is the beam size we have
        if k > 1:
            assert not stochastic, 'Beam search does not support stochastic sampling'

        sample = []
        sample_score = []
        if stochastic:
            sample_score = 0

        live_k = 1
        dead_k = 0

        hyp_samples = [[]] * live_k
        hyp_scores = np.zeros(live_k).astype(fX)
        hyp_states = []

        # get initial state of decoder rnn and encoder context
        ret = f_init(x)
        next_state, ctx0 = ret[0], ret[1]
        next_w = -1 * np.ones((1,), dtype='int64')  # bos indicator
        next_state = np.tile(next_state[None, :, :], (self.O['n_decoder_layers'], 1, 1))
        next_memory = np.zeros((self.O['n_decoder_layers'], next_state.shape[1], next_state.shape[2]), dtype=fX)

        for ii in xrange(maxlen):
            ctx = np.tile(ctx0, [live_k, 1])
            inps = [next_w, ctx, next_state]
            if unit == 'lstm':
                inps.append(next_memory)

            ret = f_next(*inps)
            next_p, next_w, next_state = ret[0], ret[1], ret[2]
            if unit == 'lstm':
                next_memory = ret[3]

            if stochastic:
                if argmax:
                    nw = next_p[0].argmax()
                else:
                    nw = next_w[0]
                sample.append(nw)
                sample_score -= np.log(next_p[0, nw])
                if nw == 0:
                    break
            else:
                cand_scores = hyp_scores[:, None] - np.log(next_p)
                cand_flat = cand_scores.flatten()
                ranks_flat = cand_flat.argsort()[:(k - dead_k)]

                voc_size = next_p.shape[1]
                trans_indices = ranks_flat / voc_size
                word_indices = ranks_flat % voc_size
                costs = cand_flat[ranks_flat]

                new_hyp_samples = []
                new_hyp_scores = np.zeros(k - dead_k).astype(fX)
                new_hyp_states = []
                new_hyp_memories = []

                for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                    new_hyp_samples.append(hyp_samples[ti] + [wi])
                    new_hyp_scores[idx] = copy.copy(costs[idx])
                    new_hyp_states.append(copy.copy(next_state[:, ti, :]))
                    new_hyp_memories.append(copy.copy(next_memory[:, ti, :]))

                # check the finished samples
                new_live_k = 0
                hyp_samples = []
                hyp_scores = []
                hyp_states = []
                hyp_memories = []

                for idx in xrange(len(new_hyp_samples)):
                    if new_hyp_samples[idx][-1] == 0:
                        sample.append(new_hyp_samples[idx])
                        sample_score.append(new_hyp_scores[idx])
                        dead_k += 1
                    else:
                        new_live_k += 1
                        hyp_samples.append(new_hyp_samples[idx])
                        hyp_scores.append(new_hyp_scores[idx])
                        hyp_states.append(new_hyp_states[idx])
                        hyp_memories.append(new_hyp_memories[idx])
                hyp_scores = np.array(hyp_scores)
                live_k = new_live_k

                if new_live_k < 1:
                    break
                if dead_k >= k:
                    break

                next_w = np.array([w[-1] for w in hyp_samples])
                next_state = np.concatenate([xx[:, None, :] for xx in hyp_states], axis=1)
                next_memory = np.concatenate([xx[:, None, :] for xx in hyp_memories], axis=1)

        if not stochastic:
            # dump every remaining one
            if live_k > 0:
                for idx in xrange(live_k):
                    sample.append(hyp_samples[idx])
                    sample_score.append(hyp_scores[idx])

        return sample, sample_score

    # Methods to build the each component of the model

    @staticmethod
    def get_input():
        """Get model input.

        Model input shape: #words * #samples

        :return: 4 Theano variables:
            x, x_mask, y, y_mask
        """

        x = T.matrix('x', dtype='int64')
        x_mask = T.matrix('x_mask', dtype=fX)
        y = T.matrix('y', dtype='int64')
        y_mask = T.matrix('y_mask', dtype=fX)

        return x, x_mask, y, y_mask

    @staticmethod
    def input_dimensions(x, y):
        """Get input dimensions.

        :param x: input x
        :param y: input y
        :return: 3 Theano variables:
            n_timestep, n_timestep_tgt, n_samples
        """

        n_timestep = x.shape[0]
        n_timestep_tgt = y.shape[0]
        n_samples = x.shape[1]

        return n_timestep, n_timestep_tgt, n_samples

    @staticmethod
    def reverse_input(x, x_mask):
        return x[::-1], x_mask[::-1]

    def embedding(self, input_, n_timestep, n_samples, emb_name='Wemb'):
        """Embedding layer: input -> embedding"""

        emb = self.P[emb_name][input_.flatten()]
        emb = emb.reshape([n_timestep, n_samples, self.O['dim_word']])

        return emb

    @staticmethod
    def dropout(input_, use_noise, trng, dropout_rate=0.5):
        """Dropout"""

        projection = T.switch(
            use_noise,
            input_ * trng.binomial(input_.shape, p=(1. - dropout_rate), n=1,
                                   dtype=input_.dtype),
            input_ * (1. - dropout_rate))
        return projection

    def feed_forward(self, input_, prefix, activation=tanh):
        """Feed-forward layer."""

        if isinstance(activation, (str, unicode)):
            activation = eval(activation)
        return activation(T.dot(input_, self.P[_p(prefix, 'W')]) + self.P[_p(prefix, 'b')])

    @staticmethod
    def get_context_mean(context, x_mask):
        """Get mean of context (across time) as initial state of decoder RNN

        Or you can use the last state of forward + backward encoder RNNs
            # return concatenate([proj[0][-1], projr[0][-1]], axis=proj[0].ndim-2)
        """

        return (context * x_mask[:, :, None]).sum(0) / x_mask.sum(0)[:, None]

    def encoder(self, src_embedding, src_embedding_r, x_mask, xr_mask, dropout_params=None):
        """GRU encoder layer: source embedding -> encoder context
        
        :return Context vector: Theano tensor
            Shape: ([Ts], [BS], [Hc])
        """

        unit = self.O['unit']
        n_layers = self.O['n_encoder_layers']
        residual = self.O['residual_enc']
        use_zigzag = self.O['use_zigzag']

        input_ = src_embedding
        input_r = src_embedding_r

        # List of inputs and outputs of each layer (for residual)
        inputs = []
        outputs = []

        # First layer (bidirectional)
        inputs.append((input_, input_r))

        h_last = get_build(unit)(self.P, inputs[-1][0], self.O, prefix='encoder', mask=x_mask, layer_id=0,
                                 dropout_params=dropout_params)[0]
        h_last_r = get_build(unit)(self.P, inputs[-1][1], self.O, prefix='encoder_r', mask=xr_mask,
                                   layer_id=0, dropout_params=dropout_params)[0]

        if self.O['encoder_many_bidirectional']:
            # First layer output
            outputs.append((h_last, h_last_r))

            # Other layers (bidirectional)
            for layer_id in xrange(1, n_layers):
                if layer_id == 1:
                    inputs.append(outputs[-1])
                else:
                    if residual == 'layer_wise':
                        # output of layer before + input of layer before
                        inputs.append((
                            outputs[-1][0] + inputs[-1][0],
                            outputs[-1][1] + inputs[-1][1],
                        ))
                    elif residual == 'last' and layer_id == n_layers - 1:
                        # only for last layer:
                        # output of layer before + mean of inputs of all layers before (except layer 0)
                        inputs.append((
                            outputs[-1][0] + average([inputs[i][0] for i in xrange(1, len(inputs))]),
                            outputs[-1][1] + average([inputs[i][1] for i in xrange(1, len(inputs))]),
                        ))
                    else:
                        inputs.append(outputs[-1])

                # Zig-zag
                x_mask_, xr_mask_ = x_mask, xr_mask
                if use_zigzag:
                    inputs[-1] = (inputs[-1][0][::-1], inputs[-1][1][::-1])
                    if layer_id % 2 == 1:
                        x_mask_, xr_mask_ = xr_mask, x_mask

                h_last = get_build(unit)(self.P, inputs[-1][0], self.O, prefix='encoder', mask=x_mask_,
                                         layer_id=layer_id, dropout_params=dropout_params)[0]
                h_last_r = get_build(unit)(self.P, inputs[-1][1], self.O, prefix='encoder_r', mask=xr_mask_,
                                           layer_id=layer_id, dropout_params=dropout_params)[0]

                outputs.append((h_last, h_last_r))

            # Context will be the concatenation of forward and backward RNNs
            if use_zigzag and n_layers % 2 == 0:
                context = concatenate([outputs[-1][0][::-1], outputs[-1][1]], axis=h_last.ndim - 1)
            else:
                context = concatenate([outputs[-1][0], outputs[-1][1][::-1]], axis=h_last.ndim - 1)
        else:
            # First layer output
            h_last = concatenate([h_last, h_last_r[::-1]], axis=h_last.ndim - 1)

            outputs.append(h_last)

            # Other layers (forward)
            for layer_id in xrange(1, n_layers):
                if layer_id == 1:
                    inputs.append(outputs[-1])
                else:
                    if residual == 'layer_wise':
                        # output of layer before + input of layer before
                        inputs.append(outputs[-1] + inputs[-1])
                    elif residual == 'last' and layer_id == n_layers - 1:
                        # only for last layer:
                        # output of layer before + mean of inputs of all layers before (except layer 0)
                        inputs.append(outputs[-1] + average(inputs[1:]))
                    else:
                        inputs.append(outputs[-1])

                x_mask_ = x_mask
                if use_zigzag:
                    inputs[-1] = inputs[-1][::-1]
                    if layer_id % 2 == 1:
                        x_mask_ = xr_mask

                # FIXME: mask modified from None to x_mask
                h_last = get_build(self.O['unit'])(self.P, inputs[-1], self.O, prefix='encoder', mask=x_mask_,
                                                   layer_id=layer_id, dropout_params=dropout_params)[0]

                outputs.append(h_last)

            # Zig-zag
            if use_zigzag and n_layers % 2 == 0:
                context = outputs[-1][::-1]
            else:
                context = outputs[-1]

        return context

    def decoder(self, tgt_embedding, y_mask, init_state, context, x_mask,
                dropout_params=None, one_step=False, init_memory=None):
        """Multi-layer GRU decoder.

        :return Decoder context vector and hidden states
        """

        n_layers = self.O['n_decoder_layers']
        unit = self.O['unit']
        attention_layer_id = self.O['attention_layer_id']
        residual = self.O['residual_dec']

        # List of inputs and outputs of each layer (for residual)
        inputs = []
        outputs = []
        hiddens_without_dropout = []
        memory_outputs = []

        # Return many things in a dict.
        kw_ret = {
            'hiddens_without_dropout': hiddens_without_dropout,
            'memory_outputs': memory_outputs,
        }

        # FIXME: init_state and init_memory
        # In training mode (one_step is False), init_state and init_memory are single state (and often None),
        #   each layer just use a copy of them.
        # In sample mode (one_step is True), init_state and init_memory are list of states,
        #   each layer use the state of its index.
        if not one_step:
            init_state = [init_state for _ in xrange(n_layers)]
            init_memory = [init_memory for _ in xrange(n_layers)]

        # Layers before attention layer
        for layer_id in xrange(0, attention_layer_id):
            # [NOTE] Do not add residual on layer 0 and 1
            if layer_id == 0:
                inputs.append(tgt_embedding)
            elif layer_id == 1:
                inputs.append(outputs[-1])
            else:
                if residual == 'layer_wise':
                    # output of layer before + input of layer before
                    inputs.append(outputs[-1] + inputs[-1])
                elif residual == 'last' and layer_id == n_layers - 1:
                    # only for last layer:
                    # output of layer before + mean of inputs of all layers before (except layer 0)
                    inputs.append(outputs[-1] + average(inputs[1:]))
                else:
                    inputs.append(outputs[-1])

            layer_out = get_build(unit)(
                self.P, inputs[-1], self.O, prefix='decoder', mask=y_mask, layer_id=layer_id,
                dropout_params=dropout_params, one_step=one_step, init_state=init_state[layer_id], context=None,
                init_memory=init_memory[layer_id],
            )

            hiddens_without_dropout.append(layer_out[-1]['hidden_without_dropout'])
            if unit == 'lstm':
                memory_outputs.append(layer_out[1])

            outputs.append(layer_out[0])

        # Attention layer
        if attention_layer_id == 0:
            inputs.append(tgt_embedding)
        elif attention_layer_id == 1:
            inputs.append(outputs[-1])
        else:
            if residual == 'layer_wise':
                inputs.append(outputs[-1] + inputs[-1])
            elif residual == 'last' and attention_layer_id == n_layers - 1:
                # only for last layer:
                # output of layer before + mean of inputs of all layers before (except layer 0)
                inputs.append(outputs[-1] + average(inputs[1:]))
            else:
                inputs.append(outputs[-1])

        hidden_decoder, context_decoder, alpha_decoder, kw_ret_att = get_build(unit + '_cond')(
            self.P, inputs[-1], self.O, prefix='decoder', mask=y_mask, context=context,
            context_mask=x_mask, one_step=one_step, init_state=init_state[attention_layer_id],
            dropout_params=dropout_params, layer_id=attention_layer_id, init_memory=init_memory[attention_layer_id],
        )

        hiddens_without_dropout.append(kw_ret_att['hidden_without_dropout'])
        if unit == 'lstm':
            memory_outputs.append(kw_ret_att['memory_output'])

        outputs.append(hidden_decoder)

        # Layers after attention layer
        for layer_id in xrange(attention_layer_id + 1, n_layers):
            if layer_id <= 1:
                inputs.append(outputs[-1])
            else:
                if residual == 'layer_wise':
                    inputs.append(outputs[-1] + inputs[-1])
                elif residual == 'last' and layer_id == n_layers - 1:
                    # only for last layer:
                    # output of layer before + mean of inputs of all layers before (except layer 0)
                    inputs.append(outputs[-1] + average(inputs[1:]))
                else:
                    inputs.append(outputs[-1])

            layer_out = get_build(unit)(
                self.P, inputs[-1], self.O, prefix='decoder', mask=y_mask, layer_id=layer_id,
                dropout_params=dropout_params, context=context_decoder, init_state=init_state[layer_id],
                one_step=one_step, init_memory=init_memory[layer_id],
            )

            hiddens_without_dropout.append(layer_out[-1]['hidden_without_dropout'])
            if unit == 'lstm':
                memory_outputs.append(layer_out[1])

            outputs.append(layer_out[0])

        return outputs[-1], context_decoder, alpha_decoder, kw_ret

    def get_word_probability(self, hidden_decoder, context_decoder, tgt_embedding, **kwargs):
        """Compute word probabilities."""

        trng = kwargs.pop('trng', RandomStreams(1234))
        use_noise = kwargs.pop('use_noise', theano.shared(np.float32(0.)))

        logit_lstm = self.feed_forward(hidden_decoder, prefix='ff_logit_lstm', activation=linear)
        logit_prev = self.feed_forward(tgt_embedding, prefix='ff_logit_prev', activation=linear)
        logit_ctx = self.feed_forward(context_decoder, prefix='ff_logit_ctx', activation=linear)
        logit = T.tanh(logit_lstm + logit_prev + logit_ctx)  # n_timestep * n_sample * dim_word
        if self.O['use_dropout']:
            logit = self.dropout(logit, use_noise, trng)
        # n_timestep * n_sample * n_words
        logit = self.feed_forward(logit, prefix='ff_logit', activation=linear)
        logit_shp = logit.shape
        probs = T.nnet.softmax(logit.reshape([logit_shp[0] * logit_shp[1],
                                              logit_shp[2]]))

        return trng, use_noise, probs

    def build_cost(self, y, y_mask, probs):
        """Build the cost from probabilities and target."""

        y_flat = y.flatten()
        y_flat_idx = T.arange(y_flat.shape[0]) * self.O['n_words'] + y_flat
        cost = -T.log(probs.flatten()[y_flat_idx])
        cost = cost.reshape([y.shape[0], y.shape[1]])
        cost = (cost * y_mask).sum(0)

        return cost

    def save_whole_model(self, model_file, iteration=-1):
        # save with iteration
        if iteration == -1:
            save_filename = '{}.iter0.npz'.format(os.path.splitext(model_file)[0])
        else:
            save_filename = '{}_iter{}.iter0.npz'.format(
                os.path.splitext(model_file)[0], iteration,
            )

        print('Saving the new model at iteration {} to {}...'.format(iteration, save_filename), end='')

        # Encoder weights from new model + other weights from old model
        old_params = dict(np.load(self.O['preload']))
        old_params.update(unzip(self.P))

        np.savez(save_filename, **old_params)

        # Save options
        with open('{}.pkl'.format(save_filename), 'wb') as f:
            pkl.dump(self.O, f)

        print('Done')
        sys.stdout.flush()

    def load_whole_model(self, model_file, iteration=-1):
        if iteration == -1:
            load_filename = '{}.iter0.npz'.format(os.path.splitext(model_file)[0])
        else:
            load_filename = '{}_iter{}.iter0.npz'.format(
                os.path.splitext(model_file)[0], iteration,
            )

        for k, v in np.load(load_filename).iteritems():
            if k in self.P:
                self.P[k].set_value(v)
