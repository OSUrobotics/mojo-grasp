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
import matplotlib.pyplot as plt


class TestReplayBuffer(unittest.TestCase):

    def setUp(self):
        with open('./test_configs/episode_all.pkl','rb') as configfile:
            self.preload_data = pkl.load(configfile)
        self.replay_buffer = ReplayBufferPriority(buffer_size=4080000)
        self.replay_buffer.preload_buffer_PKL('./test_configs/episode_all.pkl')
        print('setting up the test')

    def test_preload_buffer(self):
        print('testing the preloading')
        total_transitions = sum([len(a['timestep_list'])-1 for a in self.preload_data['episode_list']])
        self.assertEqual(total_transitions, len(self.replay_buffer), 'Number of transitions different from number of transitions in pkl file')
        weight_check = [self.replay_buffer.buffer_prio[weight] == self.replay_buffer.max_prio for weight in range(len(self.replay_buffer))]
        self.assertTrue(all(weight_check),'weights not equal to maximum when loaded')
        
    def test_sample_distribution(self):
        # all loaded samples should have same priority
        pass
        #FOR NOW, NOT RUNNING. TALK TO CINDY ABOUT GOOD WAY TO CHECK
        # print('testing the sampling')
        # sampled_inds = []
        # total_transitions = sum([len(a['timestep_list'])-1 for a in self.preload_data['episode_list']])
        # for i in range(int(total_transitions)*10):
        #     # print(i)
        #     _, _, indexes = self.replay_buffer.sample_rollout(100, 1)
        #     sampled_inds.extend(indexes)
        # unique_inds, num_sampled = np.unique(sampled_inds, return_counts=True)
        # # print(len(num_sampled))
        # plt.scatter(range(len(num_sampled)), num_sampled)
        # plt.show()
        # print('length of unique_inds', len(unique_inds), 'number of transitions', total_transitions)
        # for_chis = np.zeros(total_transitions)
        # # print(total_t)
        # for_chis[0:len(num_sampled)] = num_sampled
        # print(all(for_chis== num_sampled))
        # chis, p = chisquare(for_chis)
        # print(chis,p)
        # critical_chi = chi2.ppf(0.05,total_transitions)
        # pvalue = chi2.cdf(chis, total_transitions)
        
        # # alt_pvalue = chi2.cdf(alt_chi, len(num_sampled))
        # print('pvalue',pvalue)
        # self.assertGreater(critical_chi, chis, 'chis are larger than critical, sampling is not uniform')
        # print('done testing sample dist')
        
    def test_update_priority(self):
        self.replay_buffer.update_priorities([20,11,15,1,0], [0.4,0.1,0.03,0.2,0.9])
        self.assertEqual(self.replay_buffer.buffer_prio[0], 0.9**self.replay_buffer.alpha,'priority not updated correctly')
        self.assertEqual(self.replay_buffer.buffer_prio[1], 0.2**self.replay_buffer.alpha,'priority not updated correctly')
        self.assertEqual(self.replay_buffer.buffer_prio[11], 0.1**self.replay_buffer.alpha,'priority not updated correctly')
        self.assertEqual(self.replay_buffer.buffer_prio[15], 0.03**self.replay_buffer.alpha,'priority not updated correctly')
        self.assertEqual(self.replay_buffer.buffer_prio[20], 0.4**self.replay_buffer.alpha,'priority not updated correctly')
        self.replay_buffer.update_priorities([20], [1 + self.replay_buffer.max_prio])
        self.assertEqual(self.replay_buffer.buffer_prio[20], self.replay_buffer.max_prio, 'replay buffer not set to max prio when a number greater than max prio is suggested')

    def test_add_timestep(self):
        timestep = ('s','a','r','ns','e')
        saved_ind = self.replay_buffer.idx
        self.replay_buffer.add_timestep(timestep)
        self.assertEqual(timestep,self.replay_buffer.buffer_memory[saved_ind], 'transition not added to correct location in replay buffer OR data changed')

    def test_sample_rollout_contents(self):
        print('testing the sampling')
        total_transitions = sum([len(a['timestep_list']) for a in self.preload_data['episode_list']])
        for i in range(10):
            tr, _, _ = self.replay_buffer.sample_rollout(100, 5)
            
if __name__ == '__main__':
    unittest.main()