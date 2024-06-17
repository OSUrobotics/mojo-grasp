#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 14:02:44 2023

@author: orochi
"""
import pickle as pkl
import os
import numpy as np

thedict = {'number':1,'timestep_list':
           [{'state':
             {'two_finger_gripper':
              {'joint_angles':
               {'finger0_segment0_joint':0,
                'finger0_segment1_joint':0,
                'finger1_segment0_joint':0,
                'finger1_segment1_joint':0}}}}]}
    
# direction_full_names = ['North','North East','East','South East','South','South West','West', 'North West']
directon_key = {'N':[0,0.06],'NE':[0.0424,0.0424],'E':[0.06,0],'SE':[0.0424,-0.0424],'S':[0,-0.06],'SW':[-0.0424,-0.0424],'W':[-0.06,0], 'NW':[-0.0424,0.0424],
                'N2':[0,0.03],'NE2':[0.0424/2,0.0424/2],'E2':[0.03,0],'SE2':[0.0424/2,-0.0424/2],'S2':[0,-0.03],'SW2':[-0.0424/2,-0.0424/2],'W2':[-0.03,0], 'NW2':[-0.0424/2,0.0424/2]}
direction_full_names = ['N','NE','E','SE','S','SW','W', 'NW']
filepath = '/home/mothra/mojo-grasp/demos/rl_demo/data/Mothra_Slide/JA_S1/Real_A/'
names = os.listdir(filepath)
for ane in names:
    print('opening', ane)
    used_key = ane.split('_')[0]
    with open(filepath + ane,'rb') as fff:
        data = pkl.load(fff)
    new_data = {'number':1,'timestep_list':
               []}
    for ts in data:
        
        if type(ts) is not dict:
            print('here')
            
        else:
            print(ts.keys(), type(ts))
            new_data['timestep_list'].append({'state':
            {'two_finger_gripper':
            {'joint_angles':
                {'finger0_segment0_joint':ts['angles']['joint_1'],
                'finger0_segment1_joint':ts['angles']['joint_2'],
                'finger1_segment0_joint':ts['angles']['joint_3'],
                'finger1_segment1_joint':ts['angles']['joint_4']}},
                'obj_2':{'pose':[ts['obj_pos'],ts['obj_or']]},
            'goal_pose':{'goal_pose':directon_key[used_key]}},
            'reward':{'goal_position':[directon_key[used_key][0],directon_key[used_key][1]+0.1],
                      'distance_to_goal':np.linalg.norm(np.array(directon_key[used_key])+np.array([0,0.1])-np.array(ts['obj_pos'][0:2]))}})
            print(new_data['timestep_list'][-1])
    # input(new_data['timestep_list'])
    with open('/home/mothra/mojo-grasp/demos/rl_demo/data/Mothra_Slide/JA_S1/Real_A/trimmed'+ane,'wb') as file:
        pkl.dump(new_data,file)