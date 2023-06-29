#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 14:02:44 2023

@author: orochi
"""
import pickle as pkl


thedict = {'number':1,'timestep_list':
           [{'state':
             {'two_finger_gripper':
              {'joint_angles':
               {'finger0_segment0_joint':0,
                'finger0_segment1_joint':0,
                'finger1_segment0_joint':0,
                'finger1_segment1_joint':0}}}}]}
    
# direction_full_names = ['North','North East','East','South East','South','South West','West', 'North West']

direction_full_names = ['N','NE','E','SE','S','SW','W', 'NW']
for ane in direction_full_names:
    with open(ane+'.pkl','rb') as fff:
        data = pkl.load(fff)
    new_data = {'number':1,'timestep_list':
               []}
    for ts in data:
        print(ts)
        new_data['timestep_list'].append({'state':
          {'two_finger_gripper':
           {'joint_angles':
            {'finger0_segment0_joint':ts['angles']['joint_1'],
             'finger0_segment1_joint':ts['angles']['joint_2'],
             'finger1_segment0_joint':ts['angles']['joint_3'],
             'finger1_segment1_joint':ts['angles']['joint_4']}},
              'obj_2':{'pose':[[ts['obj_pos']],[ts['obj_or']]]}}})
        print(new_data['timestep_list'][-1])
    with open('Asterisk'+ane+'.pkl','wb') as file:
        pkl.dump(new_data,file)