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
from . import reward_base


class Reward(reward_base.RewardBase):
    _sim = None

    def __init__(self, json_path=os.path.dirname(__file__) + '/reward.json'):
        """ """
        super().__init__(json_path)

        for i in self.reward_weights.keys():
            self.reward[i] = 0
            if i == 'finger':
                self.finger_state = state_space.StateSpace(self.reward_params['Finger_State'])
            elif i == 'lift':
                self.heights = StatsTrackerBase(-0.005, 0.3)

    def get_finger_reward(self):
        return 0

    def get_lift_reward(self):
        return 0


if __name__ == "__main__":
    r = Reward()
    r.get_reward()
