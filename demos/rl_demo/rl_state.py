#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 10:33:32 2022

@author: orochi
"""

from mojograsp.simcore.state import StateDefault
import numpy as np
import pybullet as p

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
    
    def set_state(self):
        """
        Default method that sets self.current_state to either get_data() for the object or an empty dictionary
        """
        super().set_state()
        
        # if self.objects:
        #     data_dict = {}
        #     for i in self.objects:
        #         data_dict[i.name] = i.get_data()
        #     self.current_state = data_dict
        # else:
        #     self.current_state = {}

        self.current_state['f1_obj_dist'] = p.getClosestPoints(self.objects[1].id, self.objects[0].id, 1, -1, 1, -1)[0][8]
        self.current_state['f2_obj_dist'] = p.getClosestPoints(self.objects[1].id, self.objects[0].id, 1, -1, 3, -1)[0][8]