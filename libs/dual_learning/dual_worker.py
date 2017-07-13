#! /usr/bin/python
# -*- coding: utf-8 -*-

import theano
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from ..model import build_and_init_model

__author__ = 'fyabc'


class DualLearningWorker(object):
    def __init__(self, options,
                 A2B_model_path, B2A_model_path,
                 ):
        self.O = options

        # For ease of reference ,let ``A'' and ``B'' denote ``primal'' and ``dual'' models respectively
        print 'Building primal and dual model...',
        self.model_A2B = build_and_init_model(A2B_model_path, build=False)
        self.model_B2A = build_and_init_model(B2A_model_path, build=False)
        print 'Done'

        print 'Building LM...',
        self._load_LMs(self.O['LM_paths'])
        print 'Done'

        if self.O['dual_style'] == 'DSL':
            self._build_DualSL_worker()
        elif self.O['dual_style'] == 'DS3L':
            self._build_DualS3L_worker()
            self.build_samplers()
        else:
            raise Exception('Do not find a specific dual_style')

    def _load_LMs(self, LM_paths=None):
        # todo

        def _load_LM(LM_path):
            pass
        pass

    def _build_DualSL_worker(self):
        # todo
        pass

    def _build_DualSSL_worker(self):
        raise NotImplementedError('Not implemented yet')

    def _build_DualS3L_worker(self):
        # todo
        pass

    def build_samplers(self):
        trng = RandomStreams(12345)
        use_noise = theano.shared(0.)

        f_init_A2B, f_next_A2B = self.model_A2B.build_sampler(trng=trng, use_noise=use_noise, batch_mode=True)
        f_init_B2A, f_next_B2A = self.model_B2A.build_sampler(trng=trng, use_noise=use_noise, batch_mode=True)

        self.beam_search_sample_AB = lambda x, x_mask, beamSearchSize, sampleMaxlen: \
            self.model_A2B.gen_batch_sample(f_init_A2B, f_next_A2B, x, x_mask, trng,
                                            k=beamSearchSize, maxlen=sampleMaxlen, eos_id=0)

        self.beam_search_sample_BA = lambda x, x_mask, beamSearchSize, sampleMaxlen: \
            self.model_B2A.gen_batch_sample(f_init_B2A, f_next_B2A, x, x_mask, trng,
                                            k=beamSearchSize, maxlen=sampleMaxlen, eos_id=0)
