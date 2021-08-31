#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 10:56:40 2021
@author: orochi
"""
import pickle
import numpy as np
import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mojograsp.simcore.datacollection.stats_tracker_base import *
from mojograsp.simcore.simmanager.State import state_space
from collections import OrderedDict


class RewardBase:

    _sim = None

    def __init__(self,json_path=os.path.dirname(__file__)+'/reward.json'):
        """ """
        with open(json_path) as f:
            json_data = json.load(f)
        self.reward_params = json_data['Parameters'] # this is used to set any weights or paths in the individual finger/lift/grasp rewards
        self.reward_weights = json_data['Reward']

        self.reward = {}

    def get_reward(self):
        reward = 0
        for i in self.reward_weights.keys():
            try:
                self.reward[i] = eval('self.get_' + i + '_reward()') * self.reward_weights[i]
                reward += self.reward[i]
            except AttributeError:
                print("get reward method not defined for this reward: {}".format(i))
                # raise AttributeError
                self.reward[i] = None
        return reward, self.reward

    # def get_finger_reward(self):
    #     return 0
    #
    # def get_lift_reward(self):
    #     return 0
    #
    # def get_grasp_reward(self):
    #     return 0


if __name__ == "__main__":
    r = RewardBase()
    print(r.get_reward())
