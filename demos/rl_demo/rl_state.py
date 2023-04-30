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

    def __init__(self, objects: list = None, prev_len=0):
        """
        Default placeholder constructor optionally takes in a list of objects, if no list is provided it defaults
        to None. 

        :param objects: list of mojograsp objects.
        :type objects: list
        """
        super().__init__()
        self.objects = objects 
        if prev_len > 0:            
            self.previous_states = [{}]*prev_len
            self.pflag = True
        else:
            self.pflag = False
            
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
        if self.pflag:
            self.previous_states[1:] = self.previous_states[0:-1]
            self.previous_states[0] = self.current_state.copy()
        super().set_state()
        # print('Setting the state. should only see 1 per timestep')
        # if self.objects:
        #     data_dict = {}0
        #     for i in self.objects:
        #         data_dict[i.name] = i.get_data()
        #     self.current_state = data_dict
        # else:
        #     self.current_state = {}
        # print(self.objects[0].id)
        # print(self.objects[1].id)
        # print(p.getClosestPoints(self.objects[1].id, self.objects[0].id, 10, -1, 1, -1))
        # temp1 = p.getClosestPoints(self.objects[1].id, self.objects[0].id, 10, -1, 1, -1)[0]
        # temp2 = p.getClosestPoints(self.objects[1].id, self.objects[0].id, 10, -1, 4, -1)[0]
        link1_pose = p.getLinkState(self.objects[0].id, 2)
        # p.get
        link2_pose = p.getLinkState(self.objects[0].id, 5)
        link1_base = p.getLinkState(self.objects[0].id, 1)
        # p.get
        link2_base = p.getLinkState(self.objects[0].id, 4)
        # print(list(link1_pose[0]))
        # print(list(temp1[6]))
        self.current_state['f1_pos'] = list(link1_pose[0])
        self.current_state['f2_pos'] = list(link2_pose[0])
        self.current_state['f1_base'] = list(link1_base[0])
        self.current_state['f2_base'] = list(link2_base[0])
        # self.current_state['f1_pos'] = list(temp1[6])
        # self.current_state['f2_pos'] = list(temp2[6])
        # self.current_state['f1_obj_dist'] = temp1[8]
        # self.current_state['f2_obj_dist'] = temp2[8]
        # print(self.current_state['f1_obj_dist'], self.current_state['f2_obj_dist'])
        # print('set state')
        
    def init_state(self):
        """
        Default method that sets self.current_state to either get_data() for the object or an empty dictionary
        """
        super().set_state()

        # temp1 = p.getClosestPoints(self.objects[1].id, self.objects[0].id, 10, -1, 1, -1)[0]
        # temp2 = p.getClosestPoints(self.objects[1].id, self.objects[0].id, 10, -1, 4, -1)[0]
        link1_pose = p.getLinkState(self.objects[0].id, 2)
        link2_pose = p.getLinkState(self.objects[0].id, 5)
        link1_base = p.getLinkState(self.objects[0].id, 1)
        link2_base = p.getLinkState(self.objects[0].id, 4)
        self.current_state['f1_pos'] = list(link1_pose[0])
        self.current_state['f2_pos'] = list(link2_pose[0])
        self.current_state['f1_base'] = list(link1_base[0])
        self.current_state['f2_base'] = list(link2_base[0])
        # self.current_state['f1_obj_dist'] = temp1[8]
        # self.current_state['f2_obj_dist'] = temp2[8]
        if self.pflag:
            for i in range(len(self.previous_states)):
                self.previous_states[i] = self.current_state.copy()
        # print('initialized state')
         
    def get_state(self) -> dict:
        """
        Default method will return a dictionary containing the the get_data() return value for every object
        in the objects list. If no objects are given then it returns an empty dictionary.

        :return: Dictionary containing the representation of the current simulator state or an empty dictionary.
        :rtype: dict
        """
        temp = self.current_state.copy()
        if self.pflag:
            temp['previous_state'] = self.previous_states.copy()
        return temp
    
    
    def __eq__(self, o):
        # Doesnt check that the objects are the same or that the run number is the same,
        # only checks that the values saved in state are the same
        if isinstance(o, StateRL):
            return self.current_state == o.current_state
        return False