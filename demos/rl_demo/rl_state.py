#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 10:33:32 2022

@author: orochi
"""

from mojograsp.simcore.state import StateDefault
import numpy as np
import pybullet as p
from copy import deepcopy

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
    
    def __len__(self):
        return len(self.pose)
    
class RandomGoalHolder(GoalHolder):
    def __init__(self, radius_range: list):
        self.name = 'goal_pose'
        self.rrange = radius_range
        self.pose = []
        self.next_run()
        
    
    def next_run(self):
        l = np.sqrt(np.random.uniform(self.rrange[0]**2,self.rrange[1]**2))
        ang = np.pi * np.random.uniform(0,2)
        self.pose = [l * np.cos(ang),l * np.sin(ang)]
    
    def get_data(self):
        return {'goal_pose':self.pose}
        
        
class StateRL(StateDefault):
    """
    Default State Class that is used when the user does not need or wish to use the Action class
    but it may still be called by other classes such as :func:`~mojograsp.simcore.record_data.RecordData` 
    and :func:`~mojograsp.simcore.replay_buffer.ReplayBufferDefault`.
    """

    def __init__(self, objects: list = None, prev_len=0, eval_goals:GoalHolder =None):
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
        if eval_goals is not None:
            self.eval_goals = eval_goals
            self.train_goals = deepcopy(self.objects[-1])
            self.train_flag = True
        else:
            self.eval_goals = None
            
    def evaluate(self):
        if (self.eval_goals is not None) and self.train_flag:
            self.train_flag = False
            self.objects[-1] = self.eval_goals
            print('did an evaluate', self.eval_goals.pose[1],self.train_goals.pose[1])
            
    def train(self):
        if (self.eval_goals is not None) and not self.train_flag:
            self.train_flag = True
            self.objects[-1] = self.train_goals
            print('did a train', self.train_goals.pose[1])
            
    def next_run(self):
        for thing in self.objects:
            if (type(thing) == GoalHolder) | (type(thing) == RandomGoalHolder):
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
        # print('setting state')
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
        temp1 = p.getClosestPoints(self.objects[1].id, self.objects[0].id, 10, -1, 1, -1)[0]
        temp2 = p.getClosestPoints(self.objects[1].id, self.objects[0].id, 10, -1, 4, -1)[0]
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
        self.current_state['f1_ang'] = self.current_state['two_finger_gripper']['joint_angles']['finger0_segment0_joint'] + self.current_state['two_finger_gripper']['joint_angles']['finger0_segment1_joint']
        self.current_state['f2_ang'] = self.current_state['two_finger_gripper']['joint_angles']['finger1_segment0_joint'] + self.current_state['two_finger_gripper']['joint_angles']['finger1_segment1_joint']        
        self.current_state['f1_contact_pos'] = list(temp1[6])
        self.current_state['f2_contact_pos'] = list(temp2[6])
        # self.current_state['f1_obj_dist'] = temp1[8]
        # self.current_state['f2_obj_dist'] = temp2[8]
        # print(self.current_state['f1_obj_dist'], self.current_state['f2_obj_dist'])
        # print('set state')
        # print(self.current_state['f1_contact_pos'], self.current_state['f1_pos'])
        # print(self.current_state['f2_contact_pos'], self.current_state['f2_pos'])
        
    def init_state(self):
        """
        Default method that sets self.current_state to either get_data() for the object or an empty dictionary
        """
        super().set_state()
        # print('initializing state', self.pflag)
        temp1 = p.getClosestPoints(self.objects[1].id, self.objects[0].id, 10, -1, 1, -1)[0]
        temp2 = p.getClosestPoints(self.objects[1].id, self.objects[0].id, 10, -1, 4, -1)[0]
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
        self.current_state['f1_ang'] = self.current_state['two_finger_gripper']['joint_angles']['finger0_segment0_joint'] + self.current_state['two_finger_gripper']['joint_angles']['finger0_segment1_joint']
        self.current_state['f2_ang'] = self.current_state['two_finger_gripper']['joint_angles']['finger1_segment0_joint'] + self.current_state['two_finger_gripper']['joint_angles']['finger1_segment1_joint']
        self.current_state['f1_contact_pos'] = list(temp1[6])
        self.current_state['f2_contact_pos'] = list(temp2[6])
        if self.pflag:
            # print('going through to copy previous states')
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
        # print('g state')
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