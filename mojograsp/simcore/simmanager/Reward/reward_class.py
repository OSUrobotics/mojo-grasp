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


class Reward():
    _sim = None

    def __init__(self, json_path='reward.json'):
        """ """
        with open(json_path) as f:
            json_data = json.load(f)
        reward_params = json_data['Parameters']
        self.reward_weights = json_data['Reward']

        self.reward = {}

        for i in self.reward_weights.keys():
            self.reward[i] = 0
            if i == 'finger':
                self.finger_state = state_space.StateSpace(reward_params['Finger_State'])
            elif i == 'grasp':
                self.grasp_net = pickle.load(open(reward_params['Grasp_Classifier'], "rb"))
                self.grasp_reward = False
                self.grasp_state = state_space.StateSpace(reward_params['Grasp_State'])
            elif i == 'lift':
                self.heights = StatsTrackerBase(-0.005, 0.3)

    def get_reward(self):
        reward = 0
        for i in self.reward.keys():
            self.reward[i] = eval('self.get_' + i + '_reward()') * self.reward_weights[i]
            reward += self.reward[i]
        return reward, self.reward

    # class FingerReward(Reward):
    def get_finger_reward(self):
        self.finger_state.update()
        finger_obj_dists = self.finger_state.get_full_arr()
        finger_reward = -np.sum((np.array(finger_obj_dists[:6])) + (np.array(finger_obj_dists[6:])))
        return finger_reward

    def get_lift_reward(self):
        obj_pose = Reward._sim.data.get_geom_xpos("object")
        self.heights.set_value(obj_pose[-1])
        # Lift reward
        lift_reward = (self.heights.value - self.heights.min_found) / 0.2
        return lift_reward

    def get_grasp_reward(self):
        # Grasp reward
        # this feeds the state into the grasp quality predictor
        self.grasp_state.update()
        state = self.grasp_state.get_full_arr()
        grasp_quality = self.grasp_net.predict(np.array(state[0:75]).reshape(1, -1))
        if (grasp_quality >= 0.3) & (not self.grasp_reward):
            grasp_reward = 1
            self.grasp_reward = True
        else:
            grasp_reward = 0.0
        return grasp_reward


if __name__ == "__main__":
    r = Reward()
    r.get_reward()


    # Ask Nigel:
    """
    1) Reward mein why are there state space updates?
    2) Why is there a separate json for finger states?
    3) Walk through of expected usage of classes. Where will state be initialized? reward? Where will it all be stored?
    
    TODO:
    Integrate keys dictionary and ids in hand geometry class and object class into keys of state space class
    (add hand and object in init of space?)
    """