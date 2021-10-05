#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 10:56:40 2021
@author: orochi
"""
from . import reward_base
import os
import sys
import mojograsp
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Reward(reward_base.RewardBase):

    def __init__(self, json_path=os.path.dirname(__file__) + '/reward_demo.json'):
        """ """
        super().__init__(json_path)

        for i in self.reward_weights.keys():
            self.reward[i] = 0
            if i == 'contact':
                self.contact_state = mojograsp.state_space.StateSpace(self.reward_params['Contact_State'])

    def get_contact_reward(self):
        contact_reward = 0
        contact_points_info = self.contact_state.update()
        for i in contact_points_info:
            if i == 10:
                contact_reward -= 5
        return contact_reward

    def get_target_pose_reward(self):

        return 0


if __name__ == "__main__":
    r = Reward()
    r.get_reward()
