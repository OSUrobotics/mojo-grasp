#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 10:33:32 2022

@author: orochi
"""

from mojograsp.simcore.state import StateDefault
import numpy as np
from copy import deepcopy
from mojograsp.simobjects.two_finger_gripper import TwoFingerGripper
from mojograsp.simcore.goal_holder import *
        
        
class MultiprocessState(StateDefault):
    """
    Default State Class that is used when the user does not need or wish to use the Action class
    but it may still be called by other classes such as :func:`~mojograsp.simcore.record_data.RecordData` 
    and :func:`~mojograsp.simcore.replay_buffer.ReplayBufferDefault`.
    """

    def __init__(self, pybullet_instance, objects: list = None, prev_len=0, physicsClientId=None,eval_goals:GoalHolder =None):
        """
        Default placeholder constructor optionally takes in a list of objects, if no list is provided it defaults
        to None. 

        :param objects: list of mojograsp objects.
        :type objects: list
        """
        super().__init__()
        self.p = pybullet_instance
        self.objects = objects 
        for object in self.objects:
            if type(object) == TwoFingerGripper:
                temp = object.link_lengths
                self.hand_params = [temp[0][0][1],temp[0][1][1],temp[1][0][1],temp[1][1][1], object.palm_width]
                self.hand_name = object.record_name
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
            # print('did an evaluate', self.eval_goals.pose[1],self.train_goals.pose[1])
            
    def train(self):
        if (self.eval_goals is not None) and not self.train_flag:
            self.train_flag = True
            self.objects[-1] = self.train_goals
            # print('did a train', self.train_goals.pose[1])
            
    def next_run(self):
        for thing in self.objects:
            if (type(thing) == GoalHolder) | (type(thing) == RandomGoalHolder) | (type(thing) == SingleGoalHolder)|(type(thing) == HRLGoalHolder):
                fingerys = thing.next_run()
                temp = thing.get_data()
        return temp, fingerys
        
    def reset(self):
        self.run_num = 0
        for thing in self.objects:
            if (type(thing) == GoalHolder) | (type(thing) == RandomGoalHolder):
                thing.reset()
    
    def get_hand_name(self):
        # print('got hand name',self.hand_name)
        return self.hand_name

    def set_state(self):
        """
        Default method that sets self.current_state to either get_data() for the object or an empty dictionary
        """
        # print('setting state')
        if self.pflag:
            self.previous_states[1:] = self.previous_states[0:-1]
            self.previous_states[0] = deepcopy(self.current_state)
            # TODO FIGURE OUT WHY THIS DAMN THING IS FUCKED
            # SPECIFICALLY WHY THE ORIENTATION UPDATES CORRECTLY BUT THE POSITION DOES NOT
            # AND MAKE SURE THE OTHER ONES ARENT FUCKED TOO
        super().set_state()

        temp1 = self.p.getClosestPoints(self.objects[1].id, self.objects[0].id, 10, -1, 1, -1)[0]
        temp2 = self.p.getClosestPoints(self.objects[1].id, self.objects[0].id, 10, -1, 4, -1)[0]
        link1_pose = self.p.getLinkState(self.objects[0].id, 2)

        link2_pose = self.p.getLinkState(self.objects[0].id, 5)
        link1_base = self.p.getLinkState(self.objects[0].id, 1)

        link2_base = self.p.getLinkState(self.objects[0].id, 4)

        self.current_state['f1_pos'] = list(link1_pose[0])
        self.current_state['f2_pos'] = list(link2_pose[0])
        self.current_state['f1_base'] = list(link1_base[0])
        self.current_state['f2_base'] = list(link2_base[0])
        self.current_state['f1_ang'] = self.current_state['two_finger_gripper']['joint_angles']['finger0_segment0_joint'] + self.current_state['two_finger_gripper']['joint_angles']['finger0_segment1_joint']
        self.current_state['f2_ang'] = self.current_state['two_finger_gripper']['joint_angles']['finger1_segment0_joint'] + self.current_state['two_finger_gripper']['joint_angles']['finger1_segment1_joint']        
        self.current_state['f1_contact_pos'] = list(temp1[6])
        self.current_state['f2_contact_pos'] = list(temp2[6])
        self.current_state['hand_params'] = self.hand_params.copy()
        # print('sim state', self.current_state['two_finger_gripper']['joint_angles'])
        # print('joint state', self.p.getJointState(self.objects[0].id,0))
        
    def init_state(self):
        """
        Default method that sets self.current_state to either get_data() for the object or an empty dictionary
        """
        super().set_state()
        # print(self.current_state)
        temp1 = self.p.getClosestPoints(self.objects[1].id, self.objects[0].id, 10, -1, 1, -1)[0]
        temp2 = self.p.getClosestPoints(self.objects[1].id, self.objects[0].id, 10, -1, 4, -1)[0]
        link1_pose = self.p.getLinkState(self.objects[0].id, 2)
        link2_pose = self.p.getLinkState(self.objects[0].id, 5)
        link1_base = self.p.getLinkState(self.objects[0].id, 1)
        link2_base = self.p.getLinkState(self.objects[0].id, 4)
        self.current_state['f1_pos'] = list(link1_pose[0])
        self.current_state['f2_pos'] = list(link2_pose[0])
        self.current_state['f1_base'] = list(link1_base[0])
        self.current_state['f2_base'] = list(link2_base[0])

        self.current_state['f1_ang'] = self.current_state['two_finger_gripper']['joint_angles']['finger0_segment0_joint'] + self.current_state['two_finger_gripper']['joint_angles']['finger0_segment1_joint']
        self.current_state['f2_ang'] = self.current_state['two_finger_gripper']['joint_angles']['finger1_segment0_joint'] + self.current_state['two_finger_gripper']['joint_angles']['finger1_segment1_joint']
        self.current_state['f1_contact_pos'] = list(temp1[6])
        self.current_state['f2_contact_pos'] = list(temp2[6])
        if self.pflag:
            for i in range(len(self.previous_states)):
                self.previous_states[i] = self.current_state.copy()
                
         
    def get_state(self) -> dict:
        """
        Default method will return a dictionary containing the the get_data() return value for every object
        in the objects list. If no objects are given then it returns an empty dictionary.

        :return: Dictionary containing the representation of the current simulator state or an empty dictionary.
        :rtype: dict
        """
        # print('g state')
        # print('goal from get state', self.current_state['goal_pose'])
        temp = self.current_state.copy()
        if self.pflag:
            temp['previous_state'] = self.previous_states.copy()
        return temp
    
    def get_name(self) -> str:
        for thing in self.objects:
            if (type(thing) == GoalHolder) | (type(thing) == RandomGoalHolder):
                return thing.get_name()
    
    def set_goal(self,goal_list):
        for thing in self.objects:
            if (type(thing) == GoalHolder) | (type(thing) == RandomGoalHolder)|(type(thing) == HRLGoalHolder):
                thing.set_pose(goal_list[0:2], goal_list[2])
    
    def get_goal(self):
        # print(self.current_state)
        # print('goal from state.get_goal',self.current_state['goal_pose'])
        try:
            return self.current_state['goal_pose']
        except KeyError:
            return [0,0]
    def __eq__(self, o):
        # Doesnt check that the objects are the same or that the run number is the same,
        # only checks that the values saved in state are the same
        if isinstance(o, MultiprocessState):
            return self.current_state == o.current_state
        return False
