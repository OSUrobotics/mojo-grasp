#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 11:34:08 2023

@author: orochi
"""
import pickle as pkl
import os
import re
import numpy as np

top_folder = '/home/orochi/Downloads/new_nigel_data/'

all_names = os.listdir(top_folder)

names = [name for name in all_names if ('MM' in name)]

with open(top_folder+'point_reachability.pkl','rb') as file:
    point_data = pkl.load(file)

pkl_nums = []
for name in names:
    temp = re.search('\d+',name)
    pkl_nums.append(int(temp[0]))
    
pkl_sort = np.argsort(pkl_nums)

for index in pkl_sort:
    print(names[index])
    goal_pos = point_data['points'][pkl_nums[index]]
    goal_pose_local = goal_pos.copy()
    goal_pose_local[1] -= 0.0067
    goal_pos[1] -= 0.1067
    
    with open(top_folder+names[index],'rb') as file:
        temp_data = pkl.load(file)
    data_dict = {'timestep_list':[],'number':pkl_nums[index]}
    for point in temp_data:
        obj_pos = point['obj_pos']
        obj_pos[1] += 0.1067
        obj_orientation = list(point['obj_or'])
        goal_dist = np.array(obj_pos[0:2]) - np.array(goal_pose_local)
        goal_dist = np.linalg.norm(goal_dist)
        # print([obj_pos,obj_orientation])
        temp_dict = {'state':{
            'obj_2':{
                'pose':[obj_pos,obj_orientation]
                },
            'two_finger_gripper':{
                'joint_angles':{'finger0_segment0_joint':point['joint_1'],
                                'finger0_segment1_joint':point['joint_2'],
                                'finger1_segment0_joint':point['joint_3'],
                                'finger1_segment1_joint':point['joint_4'],}},
            'f1_pos':point['ee1'],
            'f2_pos':point['ee2'],
            'goal_pose':{'goal_pose':goal_pos},
            },'action':{
                },'reward':{'distance_to_goal':goal_dist,'f1_dist':point['d1'],'f2_dist':point['d2'],'goal_position':goal_pose_local
                    }
            }
        
        data_dict['timestep_list'].append(temp_dict)
        
    with open('episode_'+str(pkl_nums[index])+'.pkl','wb') as file:
        pkl.dump(data_dict,file)
        print('wrote to file ',pkl_nums[index])