#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import copy

import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from .model import NMTModel, ParameterInitializer
from ..constants import fX, profile
from ..utility.utils import *
from ..layers import *

__author__ = 'fyabc'


# TODO: code paste from yingce, need test
class TrgAttnParameterInitializer(ParameterInitializer):
    def _init_trg_attn_params(self, np_parameters):
        dim = self.O['dim']
        trg_attention_layer_id = self.O['trg_attention_layer_id']
        attention_layer_id = self.O['attention_layer_id']
        all_att = self.O['decoder_all_attention']
        if all_att:
            raise Exception('Not implemented yet')
        else:
            import math
            prefix = 'decoder'

            def _gen_block():
                return np.concatenate([orthogonal_weight(dim) for _ in xrange(4)], axis=1)

            np_parameters[_p(prefix, 'Wc_trg', trg_attention_layer_id)] = _gen_block()
            np_parameters[_p(prefix, 'Wh_trgatt', trg_attention_layer_id)] = orthogonal_weight(dim)
            np_parameters[_p(prefix, 'U_trgatt', trg_attention_layer_id)] = orthogonal_weight(dim)
            np_parameters[_p(prefix, 'b_trgatt', trg_attention_layer_id)] = np.zeros((dim,), dtype='float32')
            np_parameters[_p(prefix, 'v_trgatt', trg_attention_layer_id)] = \
                np.random.rand(dim, 1).astype('float32') / math.sqrt(dim)
            np_parameters[_p(prefix, 'c_trgatt', trg_attention_layer_id)] = np.zeros((1,), dtype='float32')

            if attention_layer_id != trg_attention_layer_id:
                np_parameters[_p(prefix, 'U_nl_trg', trg_attention_layer_id)] = _gen_block()
                np_parameters[_p(prefix, 'b_nl_trg', trg_attention_layer_id)] = np.zeros((4 * dim,), dtype='float32')

            if attention_layer_id < trg_attention_layer_id:
                np_parameters[_p(prefix, 'Wc', trg_attention_layer_id)] = \
                    np.concatenate([_gen_block(), _gen_block()], axis=0)

            self.init_feed_forward(np_parameters, prefix='ff_logit_ctx_trg', nin=dim,
                                   nout=self.O['dim_word'], orthogonal=False)

    def init_params(self):
        np_parameters = super(TrgAttnParameterInitializer, self).init_params()
        self._init_trg_attn_params(np_parameters)
        return np_parameters


# TODO: code paste from yingce, need test
class TrgAttnNMTModel(NMTModel):
    def __init__(self, options, given_params=None):
        super(TrgAttnNMTModel, self).__init__(options, given_params)
        self.initializer = TrgAttnParameterInitializer(options)

    def decoder(self, tgt_embedding, y_mask, init_state, context, x_mask,
                dropout_params=None, one_step=False, init_memory=None, **kwargs):
        """Multi-layer GRU decoder.

        :return Decoder context vector and hidden states and others (kw_ret)
        """

        n_layers = self.O['n_decoder_layers']
        unit = self.O['unit']
        attention_layer_id = self.O['attention_layer_id']
        trg_attention_layer_id = self.O['trg_attention_layer_id']
        residual = self.O['residual_dec']
        all_att = self.O['decoder_all_attention']
        avg_ctx = self.O['average_context']
        # FIXME: Add get_gates for only common mode (one attention) here.
        get_gates = kwargs.pop('get_gates', False)

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

        if get_gates:
            kw_ret['input_gates'] = []
            kw_ret['forget_gates'] = []
            kw_ret['output_gates'] = []

        # FIXME: init_state and init_memory
        # In training mode (one_step is False), init_state and init_memory are single state (and often None),
        #   each layer just use a copy of them.
        # In sample mode (one_step is True), init_state and init_memory are list of states,
        #   each layer use the state of its index.
        if not one_step:
            init_state = [init_state for _ in xrange(n_layers)]
            init_memory = [init_memory for _ in xrange(n_layers)]

        if all_att:
            # All decoder layers have attention.
            context_decoder_list = []
            context_decoder = None
            alpha_decoder = None

            for layer_id in xrange(0, n_layers):
                # [NOTE] Do not add residual on layer 0 and 1
                if layer_id == 0:
                    inputs.append(tgt_embedding)
                elif layer_id == 1:
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

                hidden_decoder, context_decoder, alpha_decoder, kw_ret_att = get_build(unit + '_cond')(
                    self.P, inputs[-1], self.O, prefix='decoder', mask=y_mask, context=context,
                    context_mask=x_mask, one_step=one_step, init_state=init_state[layer_id],
                    dropout_params=dropout_params, layer_id=layer_id,
                    init_memory=init_memory[layer_id],
                )

                context_decoder_list.append(context_decoder)

                hiddens_without_dropout.append(kw_ret_att['hidden_without_dropout'])
                if 'lstm' in unit:
                    memory_outputs.append(kw_ret_att['memory_output'])

                outputs.append(hidden_decoder)

            if avg_ctx:
                ctx_output = T.mean(T.stack(context_decoder_list, axis=0), axis=0)
            else:
                ctx_output = context_decoder
            return outputs[-1], ctx_output, alpha_decoder, kw_ret

        else:
            # Layers before attention layer
            first_attn_layer = min(attention_layer_id, trg_attention_layer_id)
            second_attn_layer = max(attention_layer_id, trg_attention_layer_id)

            # predefine the two vars here to remove the warnings
            context_trg_prev = None
            alpha_decoder = None

            # deal with the layers before the first attention layer
            for layer_id in xrange(0, first_attn_layer):
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
                    init_memory=init_memory[layer_id], get_gates=get_gates,
                )
                kw_ret_layer = layer_out[-1]

                hiddens_without_dropout.append(kw_ret_layer['hidden_without_dropout'])
                if 'lstm' in unit:
                    memory_outputs.append(layer_out[1])
                if get_gates:
                    kw_ret['input_gates'].append(kw_ret_layer['input_gates'])
                    kw_ret['forget_gates'].append(kw_ret_layer['forget_gates'])
                    kw_ret['output_gates'].append(kw_ret_layer['output_gates'])

                outputs.append(layer_out[0])

            # Deal with the first attention layer
            if first_attn_layer == 0:
                inputs.append(tgt_embedding)
            elif first_attn_layer == 1:
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

            if first_attn_layer == second_attn_layer:
                hidden_decoder, context_decoder, context_trg_prev, alpha_decoder, kw_ret_att = \
                    get_build(unit + '_srctrgattn_layer')(
                        self.P, inputs[-1], self.O, prefix='decoder', mask=y_mask,
                        context=context, context_mask=x_mask, one_step=one_step,
                        init_state=init_state[first_attn_layer],
                        dropout_params=dropout_params, layer_id=first_attn_layer,
                        init_memory=init_memory[first_attn_layer],
                        get_gates=get_gates, **kwargs)
            elif first_attn_layer == attention_layer_id:
                hidden_decoder, context_decoder, alpha_decoder, kw_ret_att = get_build(unit + '_cond')(
                    self.P, inputs[-1], self.O, prefix='decoder', mask=y_mask, context=context,
                    context_mask=x_mask, one_step=one_step, init_state=init_state[attention_layer_id],
                    dropout_params=dropout_params, layer_id=attention_layer_id,
                    init_memory=init_memory[attention_layer_id],
                    get_gates=get_gates
                )
            else:
                hidden_decoder, context_trg_prev, kw_ret_att = get_build(unit + '_trgattn_beforesrc')(
                    self.P, inputs[-1], self.O, prefix='decoder', mask=y_mask, one_step=one_step,
                    init_memory=init_memory[first_attn_layer],
                    init_state=init_state[first_attn_layer],
                    dropout_params=dropout_params, layer_id=first_attn_layer,
                    get_gates=get_gates, **kwargs)

            hiddens_without_dropout.append(kw_ret_att['hidden_without_dropout'])
            if 'lstm' in unit:
                memory_outputs.append(kw_ret_att['memory_output'])
            if get_gates:
                kw_ret['input_gates'].append(kw_ret_att['input_gates'])
                kw_ret['forget_gates'].append(kw_ret_att['forget_gates'])
                kw_ret['output_gates'].append(kw_ret_att['output_gates'])
                kw_ret['input_gates_att'] = kw_ret_att['input_gates_att']
                kw_ret['forget_gates_att'] = kw_ret_att['forget_gates_att']
                kw_ret['output_gates_att'] = kw_ret_att['output_gates_att']

            outputs.append(hidden_decoder)

            # Layers after first layer
            for layer_id in xrange(first_attn_layer + 1, second_attn_layer):
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

                if attention_layer_id == first_attn_layer:
                    layer_out = get_build(unit)(
                        self.P, inputs[-1], self.O, prefix='decoder', mask=y_mask, layer_id=layer_id,
                        dropout_params=dropout_params, context=context_decoder, init_state=init_state[layer_id],
                        one_step=one_step, init_memory=init_memory[layer_id], get_gates=get_gates,
                    )
                else:
                    layer_out = get_build(unit)(
                        self.P, inputs[-1], self.O, prefix='decoder', mask=y_mask, layer_id=layer_id,
                        dropout_params=dropout_params, one_step=one_step, init_state=init_state[layer_id], context=None,
                        init_memory=init_memory[layer_id], get_gates=get_gates,
                    )
                kw_ret_layer = layer_out[-1]

                hiddens_without_dropout.append(kw_ret_layer['hidden_without_dropout'])
                if 'lstm' in unit:
                    memory_outputs.append(layer_out[1])
                if get_gates:
                    kw_ret['input_gates'].append(kw_ret_layer['input_gates'])
                    kw_ret['forget_gates'].append(kw_ret_layer['forget_gates'])
                    kw_ret['output_gates'].append(kw_ret_layer['output_gates'])

                outputs.append(layer_out[0])

            if first_attn_layer != second_attn_layer:
                if second_attn_layer == 1:
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

                if second_attn_layer == attention_layer_id:
                    hidden_decoder, context_decoder, alpha_decoder, kw_ret_att = get_build(unit + '_cond')(
                        self.P, inputs[-1], self.O, prefix='decoder', mask=y_mask, context=context,
                        context_mask=x_mask, one_step=one_step, init_state=init_state[second_attn_layer],
                        dropout_params=dropout_params, layer_id=second_attn_layer,
                        init_memory=init_memory[second_attn_layer],
                        get_gates=get_gates,
                    )
                else:
                    hidden_decoder, context_trg_prev, kw_ret_att = get_build(unit + '_trgattn_aftersrc')(
                        self.P, inputs[-1], self.O, prefix='decoder', mask=y_mask, one_step=one_step,
                        init_memory=init_memory[second_attn_layer],
                        init_state=init_state[second_attn_layer],
                        dropout_params=dropout_params, layer_id=second_attn_layer,
                        context_src=context_decoder,
                        get_gates=get_gates, **kwargs)

                hiddens_without_dropout.append(kw_ret_att['hidden_without_dropout'])
                if 'lstm' in unit:
                    memory_outputs.append(kw_ret_att['memory_output'])
                if get_gates:
                    kw_ret['input_gates'].append(kw_ret_att['input_gates'])
                    kw_ret['forget_gates'].append(kw_ret_att['forget_gates'])
                    kw_ret['output_gates'].append(kw_ret_att['output_gates'])
                    kw_ret['input_gates_att'] = kw_ret_att['input_gates_att']
                    kw_ret['forget_gates_att'] = kw_ret_att['forget_gates_att']
                    kw_ret['output_gates_att'] = kw_ret_att['output_gates_att']

                outputs.append(hidden_decoder)

            # Layers after attention layer
            for layer_id in xrange(second_attn_layer + 1, n_layers):
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
                    one_step=one_step, init_memory=init_memory[layer_id], get_gates=get_gates,
                )
                kw_ret_layer = layer_out[-1]

                hiddens_without_dropout.append(kw_ret_layer['hidden_without_dropout'])
                if 'lstm' in unit:
                    memory_outputs.append(layer_out[1])
                if get_gates:
                    kw_ret['input_gates'].append(kw_ret_layer['input_gates'])
                    kw_ret['forget_gates'].append(kw_ret_layer['forget_gates'])
                    kw_ret['output_gates'].append(kw_ret_layer['output_gates'])

                outputs.append(layer_out[0])

            return outputs[-1], context_decoder, context_trg_prev, alpha_decoder, kw_ret

    def get_word_probability_(self, hidden_decoder, context_decoder, context_trg_prev, tgt_embedding, **kwargs):
        """Compute word probabilities."""

        trng = kwargs.pop('trng', RandomStreams(1234))
        use_noise = kwargs.pop('use_noise', theano.shared(np.float32(0.)))

        logit_lstm = self.feed_forward(hidden_decoder, prefix='ff_logit_lstm', activation=linear)
        logit_prev = self.feed_forward(tgt_embedding, prefix='ff_logit_prev', activation=linear)
        logit_ctx = self.feed_forward(context_decoder, prefix='ff_logit_ctx', activation=linear)
        logit_ctx_trg = self.feed_forward(context_trg_prev, prefix='ff_logit_ctx_trg', activation=linear)
        logit = T.tanh(logit_lstm + logit_prev + logit_ctx + logit_ctx_trg)  # n_timestep * n_sample * dim_word
        if self.O['use_dropout']:
            logit = self.dropout(logit, use_noise, trng)
        # n_timestep * n_sample * n_words
        logit = self.feed_forward(logit, prefix='ff_logit', activation=linear)
        logit_shp = logit.shape
        probs = T.nnet.softmax(logit.reshape([-1, logit_shp[2]]))

        return trng, use_noise, probs

    def build_model(self, set_instance_variables=False):
        """Build a training model."""

        dropout_rate = self.O['use_dropout']

        opt_ret = {}

        trng = RandomStreams(1234)
        use_noise = theano.shared(np.float32(0.))

        if dropout_rate is not False:
            dropout_params = [use_noise, trng, dropout_rate]
        else:
            dropout_params = None

        (x, x_mask, y, y_mask), context, _ = self.input_to_context(dropout_params=dropout_params)

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
        hidden_decoder, context_decoder, context_trg_prev, opt_ret['dec_alphas'], _ = self.decoder(
            tgt_embedding, y_mask=y_mask, init_state=init_decoder_state, context=context, x_mask=x_mask,
            dropout_params=dropout_params, one_step=False,
        )

        trng, use_noise, probs = self.get_word_probability_(hidden_decoder, context_decoder, context_trg_prev,
                                                            tgt_embedding, trng=trng, use_noise=use_noise)

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

        if set_instance_variables:
            # Unused now
            self.x, self.x_mask, self.y, self.y_mask = x, x_mask, y, y_mask

        return trng, use_noise, x, x_mask, y, y_mask, opt_ret, cost, context_mean

    def build_sampler(self, **kwargs):
        """Build a sampler.

        :returns f_init, f_next
            f_init: Theano function
                inputs: x, [if batch mode: x_mask]
                outputs: init_state, ctx

            f_next: Theano function
                inputs: y, ctx, [if batch mode: x_mask], init_state, [if LSTM unit: init_memory]
                outputs: next_probs, next_sample, hiddens_without_dropout, [if LSTM unit: memory_out],
                    [if get_gates:
                        T.stack(kw_ret['input_gates']),
                        T.stack(kw_ret['forget_gates']),
                        T.stack(kw_ret['output_gates']),
                        kw_ret['input_gates_att'],
                        kw_ret['forget_gates_att'],
                        kw_ret['output_gates_att'],
                    ]
        """
        batch_mode = kwargs.pop('batch_mode', False)
        trng = kwargs.pop('trng', RandomStreams(1234))
        use_noise = kwargs.pop('use_noise', theano.shared(np.float32(0.)))
        get_gates = kwargs.pop('get_gates', False)

        unit = self.O['unit']

        x = T.matrix('x', dtype='int64')
        xr = x[::-1]
        n_timestep = x.shape[0]
        n_samples = x.shape[1]

        if batch_mode:
            x_mask = T.matrix('x_mask', dtype=fX)
            xr_mask = x_mask[::-1]

        # Word embedding for forward rnn and backward rnn (source)
        src_embedding = self.embedding(x, n_timestep, n_samples)
        src_embedding_r = self.embedding(xr, n_timestep, n_samples)

        # Encoder
        ctx, _ = self.encoder(
            src_embedding, src_embedding_r,
            x_mask if batch_mode else None, xr_mask if batch_mode else None,
            dropout_params=None,
        )

        # Get the input for decoder rnn initializer mlp
        ctx_mean = self.get_context_mean(ctx, x_mask) if batch_mode else ctx.mean(0)
        init_state = self.feed_forward(ctx_mean, prefix='ff_state', activation=tanh)

        print('Building f_init...', end='')
        inps = [x]
        if batch_mode:
            inps.append(x_mask)
        outs = [init_state, ctx]
        f_init = theano.function(inps, outs, name='f_init', profile=profile)
        print('Done')

        # x: 1 x 1
        y = T.vector('y_sampler', dtype='int64')
        init_state = T.tensor3('init_state', dtype=fX)
        init_memory = T.tensor3('init_memory', dtype=fX)
        current_step = T.scalar('current_step', dtype='int64')
        decoded_h = T.tensor3('decoded_h', dtype=fX)

        # If it's the first word, emb should be all zero and it is indicated by -1
        emb = T.switch(y[:, None] < 0,
                       T.alloc(0., 1, self.P['Wemb_dec'].shape[1]),
                       self.P['Wemb_dec'][y])

        # Apply one step of conditional gru with attention
        hidden_decoder, context_decoder, context_trg_prev, _, kw_ret = self.decoder(
            emb, y_mask=None, init_state=init_state, context=ctx,
            x_mask=x_mask if batch_mode else None,
            dropout_params=None, one_step=True, init_memory=init_memory,
            get_gates=get_gates,
            current_step=current_step,
            decoded_h=decoded_h
        )

        # Get memory_out and hiddens_without_dropout
        # FIXME: stack list into a single tensor
        memory_out = None
        hiddens_without_dropout = T.stack(kw_ret['hiddens_without_dropout'])
        if 'lstm' in unit:
            memory_out = T.stack(kw_ret['memory_outputs'])

        logit_lstm = self.feed_forward(hidden_decoder, prefix='ff_logit_lstm', activation=linear)
        logit_prev = self.feed_forward(emb, prefix='ff_logit_prev', activation=linear)
        logit_ctx = self.feed_forward(context_decoder, prefix='ff_logit_ctx', activation=linear)
        logit_ctx_trg = self.feed_forward(context_trg_prev, prefix='ff_logit_ctx_trg', activation=linear)
        logit = T.tanh(logit_lstm + logit_prev + logit_ctx + logit_ctx_trg)
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
        inps = [y, ctx, init_state, decoded_h, current_step]
        if batch_mode:
            inps.insert(2, x_mask)
        outs = [next_probs, next_sample, hiddens_without_dropout]
        if 'lstm' in unit:
            inps.append(init_memory)
            outs.append(memory_out)
        if get_gates:
            outs.extend([
                T.stack(kw_ret['input_gates']),
                T.stack(kw_ret['forget_gates']),
                T.stack(kw_ret['output_gates']),
                kw_ret['input_gates_att'],
                kw_ret['forget_gates_att'],
                kw_ret['output_gates_att'],
            ])
        f_next = theano.function(inps, outs, name='f_next', profile=profile)
        print('Done')

        return f_init, f_next

    def gen_sample(self, f_init, f_next, x, trng=None, k=1, maxlen=30,
                   stochastic=True, argmax=False, **kwargs):
        """Generate sample, either with stochastic sampling or beam search. Note that,

        this function iteratively calls f_init and f_next functions.
        """

        kw_ret = {}
        have_kw_ret = bool(kwargs)

        get_gates = kwargs.pop('get_gates', False)
        if get_gates:
            kw_ret['input_gates_list'] = []
            kw_ret['forget_gates_list'] = []
            kw_ret['output_gates_list'] = []
            kw_ret['input_gates_att_list'] = []
            kw_ret['forget_gates_att_list'] = []
            kw_ret['output_gates_att_list'] = []
            kw_ret['state_list'] = []
            kw_ret['memory_list'] = []

        unit = self.O['unit']
        trg_attnlayer_id = self.O['trg_attention_layer_id']
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

        print(next_state.shape)

        decoded_h_all = np.zeros((2, next_state.shape[0], self.O['dim']), dtype='float32')
        decoded_h_all[0] = next_state
        next_state = np.tile(next_state[None, :, :], (self.O['n_decoder_layers'], 1, 1))
        next_memory = np.zeros((self.O['n_decoder_layers'], next_state.shape[1], next_state.shape[2]), dtype=fX)

        for ii in xrange(maxlen):
            ctx = np.tile(ctx0, [live_k, 1])
            inps = [next_w, ctx, next_state, decoded_h_all, ii]
            if 'lstm' in unit:
                inps.append(next_memory)

            ret = f_next(*inps)
            next_p, next_w, next_state = ret[0], ret[1], ret[2]
            if 'lstm' in unit:
                next_memory = ret[3]

            if get_gates:
                kw_ret['input_gates_list'].append(ret[-6])
                kw_ret['forget_gates_list'].append(ret[-5])
                kw_ret['output_gates_list'].append(ret[-4])
                kw_ret['input_gates_att_list'].append(ret[-3])
                kw_ret['forget_gates_att_list'].append(ret[-2])
                kw_ret['output_gates_att_list'].append(ret[-1])
                kw_ret['state_list'].append(next_state)
                if 'lstm' in unit:
                    kw_ret['memory_list'].append(ret[3])

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
                new_decoder_h_vector_all_H = []
                new_decoder_h_vector_all_N = []

                for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                    new_hyp_samples.append(hyp_samples[ti] + [wi])
                    new_hyp_scores[idx] = copy.copy(costs[idx])
                    new_hyp_states.append(copy.copy(next_state[:, ti, :]))
                    new_hyp_memories.append(copy.copy(next_memory[:, ti, :]))
                    new_decoder_h_vector_all_H.append(decoded_h_all[:-1, ti, :])
                    new_decoder_h_vector_all_N.append(next_state[trg_attnlayer_id, ti, :])

                # check the finished samples
                new_live_k = 0
                hyp_samples = []
                hyp_scores = []
                hyp_states = []
                hyp_memories = []
                selected_idx = []

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
                        selected_idx.append(idx)
                hyp_scores = np.array(hyp_scores)
                live_k = new_live_k

                if new_live_k < 1:
                    break
                if dead_k >= k:
                    break

                decoded_h_all = np.zeros((ii + 3, live_k, self.O['dim']), dtype='float32')

                for jj, win_id in enumerate(selected_idx):
                    decoded_h_all[:(ii + 2), jj, :] = np.concatenate(
                        [new_decoder_h_vector_all_H[win_id], new_decoder_h_vector_all_N[win_id][None, :]],
                        axis=0)

                next_w = np.array([w[-1] for w in hyp_samples])
                next_state = np.concatenate([xx[:, None, :] for xx in hyp_states], axis=1)
                next_memory = np.concatenate([xx[:, None, :] for xx in hyp_memories], axis=1)

        if not stochastic:
            # dump every remaining one
            if live_k > 0:
                for idx in xrange(live_k):
                    sample.append(hyp_samples[idx])
                    sample_score.append(hyp_scores[idx])

        if have_kw_ret:
            return sample, sample_score, kw_ret
        return sample, sample_score

    def gen_batch_sample(self, f_init, f_next, x, x_mask, trng=None, k=1, maxlen=30, eos_id=0, **kwargs):
        """
        Only used for Batch Beam Search;
        Do not Support Stochastic Sampling
        """

        kw_ret = {}
        have_kw_ret = bool(kwargs)

        ret_memory = kwargs.pop('ret_memory', False)
        if ret_memory:
            kw_ret['memory'] = []

        unit = self.O['unit']

        batch_size = x.shape[1]
        sample = [[] for _ in xrange(batch_size)]
        sample_score = [[] for _ in xrange(batch_size)]

        lives_k = [1] * batch_size
        deads_k = [0] * batch_size

        batch_hyp_samples = [[[]] for _ in xrange(batch_size)]
        batch_hyp_scores = [np.zeros(ii, dtype=fX) for ii in lives_k]

        # get initial state of decoder rnn and encoder context
        ret = f_init(x, x_mask)
        next_state, ctx0 = ret[0], ret[1]
        next_w = np.array([-1] * batch_size, dtype='int64')  # bos indicator
        next_state = np.tile(next_state[None, :, :], (self.O['n_decoder_layers'], 1, 1))
        next_memory = np.zeros((self.O['n_decoder_layers'], next_state.shape[1], next_state.shape[2]), dtype=fX)

        for ii in xrange(maxlen):
            ctx = np.repeat(ctx0, lives_k, axis=1)
            x_extend_masks = np.repeat(x_mask, lives_k, axis=1)
            cursor_start, cursor_end = 0, lives_k[0]
            for jj in xrange(batch_size):
                if lives_k[jj] > 0:
                    ctx[:, cursor_start: cursor_end, :] = np.repeat(ctx0[:, jj, :][:, None, :], lives_k[jj], axis=1)
                    x_extend_masks[:, cursor_start: cursor_end] = np.repeat(x_mask[:, jj][:, None], lives_k[jj], axis=1)
                if jj < batch_size - 1:
                    cursor_start = cursor_end
                    cursor_end += lives_k[jj + 1]

            inps = [next_w, ctx, x_extend_masks, next_state]
            if 'lstm' in unit:
                inps.append(next_memory)

            ret = f_next(*inps)

            if 'lstm' in unit:
                next_memory = ret[3]

                if ret_memory:
                    kw_ret['memory'].append(next_memory)

            next_w_list = []
            next_state_list = []
            next_memory_list = []

            next_p, next_state = ret[0], ret[2]
            cursor_start, cursor_end = 0, lives_k[0]

            for jj in xrange(batch_size):
                if cursor_start == cursor_end:
                    if jj < batch_size - 1:
                        cursor_end += lives_k[jj + 1]
                    continue
                index_range = range(cursor_start, cursor_end)
                cand_scores = batch_hyp_scores[jj][:, None] - np.log(next_p[index_range, :])
                cand_flat = cand_scores.flatten()

                try:
                    from bottleneck import argpartition as part_sort
                    ranks_flat = part_sort(cand_flat, kth=k - deads_k[jj] - 1)[:k - deads_k[jj]]
                except ImportError:
                    from bottleneck import argpartsort as part_sort
                    ranks_flat = part_sort(cand_flat, k - deads_k[jj])[:k - deads_k[jj]]

                voc_size = next_p.shape[1]
                trans_indices = ranks_flat / voc_size
                word_indices = ranks_flat % voc_size
                costs = cand_flat[ranks_flat]

                new_hyp_samples = []
                new_hyp_scores = np.zeros(k - deads_k[jj]).astype('float32')
                new_hyp_states = []
                new_hyp_memories = []

                for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                    new_hyp_samples.append(batch_hyp_samples[jj][ti] + [wi])
                    new_hyp_scores[idx] = copy.copy(costs[idx])
                    new_hyp_states.append(copy.copy(next_state[:, cursor_start + ti, :]))
                    new_hyp_memories.append(copy.copy(next_memory[:, cursor_start + ti, :]))

                # check the finished samples
                new_live_k = 0
                batch_hyp_samples[jj] = []
                hyp_scores = []
                hyp_states = []
                hyp_memories = []

                for idx in xrange(len(new_hyp_samples)):
                    if new_hyp_samples[idx][-1] == eos_id:
                        sample[jj].append(new_hyp_samples[idx])
                        sample_score[jj].append(new_hyp_scores[idx])
                        deads_k[jj] += 1
                    else:
                        new_live_k += 1
                        batch_hyp_samples[jj].append(new_hyp_samples[idx])
                        hyp_scores.append(new_hyp_scores[idx])
                        hyp_states.append(new_hyp_states[idx])
                        hyp_memories.append(new_hyp_memories[idx])

                batch_hyp_scores[jj] = np.array(hyp_scores)
                lives_k[jj] = new_live_k

                if jj < batch_size - 1:
                    cursor_start = cursor_end
                    cursor_end += lives_k[jj + 1]

                if hyp_states:
                    next_w_list += [w[-1] for w in batch_hyp_samples[jj]]
                    next_state_list += [xx[:, None, :] for xx in hyp_states]
                    next_memory_list += [xx[:, None, :] for xx in hyp_memories]

            if np.array(lives_k).sum() > 0:
                next_w = np.array(next_w_list)
                next_state = np.concatenate(next_state_list[:], axis=1)
                next_memory = np.concatenate(next_memory_list[:], axis=1)
            else:
                break

        # dump every remaining one
        for jj in xrange(batch_size):
            if lives_k[jj] > 0:
                for idx in xrange(lives_k[jj]):
                    sample[jj].append(batch_hyp_samples[jj][idx])
                    sample_score[jj].append(batch_hyp_scores[jj][idx])

        if have_kw_ret:
            return sample, sample_score, kw_ret
        return sample, sample_score


__all__ = [
    'TrgAttnNMTModel',
    'TrgAttnParameterInitializer',
]
