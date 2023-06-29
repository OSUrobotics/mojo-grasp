#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 10:05:29 2023

@author: orochi
"""

import gym
from gym import spaces
from environment import Environment

class GymWrapper(gym.Env):
    '''
    Example environment that follows gym interface to allow us to use openai gym learning algorithms with mojograsp
    '''
    
    def __init__(self):
        super(GymWrapper,self).__init__()
        self.env = Environment()
        pass
    
    def reset(self):
        self.env.reset()
        
    def step(self, action):
        self.env.step()
        
    def render(self):
        pass
    
    def close(self):
        pass