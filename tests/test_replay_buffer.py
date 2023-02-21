#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 10:29:12 2023

@author: orochi
"""


import unittest
from mojograsp.simcore.DDPGfD import DDPGfD_priority
import pickle as pkl
import numpy as np
import torch
from mojograsp.simcore.priority_replay_buffer import ReplayBufferPriority
from scipy.stats import chisquare
from scipy.stats import chi2

class TestDDPG(unittest.TestCase):

    def setUp(self):
        with open('./test_configs/episode_all.pkl','rb') as configfile:
            self.preload_data = pkl.load(configfile)
        self.replay_buffer = ReplayBufferPriority(buffer_size=4080000)
        self.replay_buffer.preload_buffer_PKL('./test_configs/episode_all.pkl')

    def test_preload_buffer(self):
        print('testing the preloading')
        total_transitions = sum([len(a['timestep_list'])-1 for a in self.preload_data['episode_list']])
        self.assertEqual(total_transitions, len(self.replay_buffer), 'Number of transitions different from number of transitions in pkl file')
        weight_check = [self.replay_buffer.buffer_prio[weight] == self.replay_buffer.max_prio for weight in range(len(self.replay_buffer))]
        self.assertTrue(all(weight_check),'weights not equal to maximum when loaded')
        
    def test_sample_distribution(self):
        # all loaded samples should have same priority
        print('testing the sampling')
        sampled_inds = []
        total_transitions = sum([len(a['timestep_list']) for a in self.preload_data['episode_list']])
        for i in range(int(total_transitions/10)):
            # print(i)
            _, _, indexes = self.replay_buffer.sample_rollout(100, 5)
            sampled_inds.extend(indexes)
        unique_inds, num_sampled = np.unique(sampled_inds, return_counts=True)
        chis, p = chisquare(num_sampled)
        critical_chi = chi2.ppf(0.05,total_transitions)
        self.assertGreater(critical_chi, chis, 'chis are larger than critical, sampling is not uniform')
        print('done testing sample dist')
            
if __name__ == '__main__':
    unittest.main()