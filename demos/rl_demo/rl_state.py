#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 10:33:32 2022

@author: orochi
"""

from mojograsp.simcore.state import StateDefault
import numpy as np

class GoalHolder():
    def __init__(self, goal_pose):
        self.pose = goal_pose
        self.name = 'goal_pose'
        if len(np.shape(self.pose)) == 1:
            self.pose = [self.pose]
        self.run_num = 0
    
    def get_data(self):
        return {'goal_pose':self.pose[self.run_num]}

    def next_run(self):
        self.run_num +=1
        
    def reset(self):
        self.run_num = 0
        
class StateRL(StateDefault):
    """
    Default State Class that is used when the user does not need or wish to use the Action class
    but it may still be called by other classes such as :func:`~mojograsp.simcore.record_data.RecordData` 
    and :func:`~mojograsp.simcore.replay_buffer.ReplayBufferDefault`.
    """

    def __init__(self, objects: list = None):
        """
        Default placeholder constructor optionally takes in a list of objects, if no list is provided it defaults
        to None. 

        :param objects: list of mojograsp objects.
        :type objects: list
        """
        super().__init__()
        self.objects = objects    
        
    def next_run(self):
        for thing in self.objects:
            if type(thing) == GoalHolder:
                thing.next_run()
        
    def reset(self):
        self.run_num = 0
        for thing in self.objects:
            if type(thing) == GoalHolder:
                thing.reset()