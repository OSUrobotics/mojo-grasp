#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 14:47:04 2022

@author: orochi
"""
import cv2
import os
import pickle as pkl
import numpy as np

def label_images(image_path):
    font = cv2.FONT_HERSHEY_SIMPLEX
    all_names = os.listdir(image_path)
    image_names = []
    pkl_names =[]
    for name in all_names:
        if "Evaluation" in name:
            if '.png' in name:
                image_names.append(name)
            elif '.pkl' in name:
                pkl_names.append(name)
    # print(image_names)
    imlist = []
    data_list = {}
    for name in pkl_names:
        with open(image_path + name, 'rb') as datafile: 
            temp = pkl.load(datafile)
            data_list[temp['number']] = temp.copy()
    # print(data_list.keys())
    print(f'processing {len(image_names)} total images')
    for imcount,name in enumerate(image_names):
        if imcount % 100 == 0:
            print(f'processed {imcount/len(image_names) * 100}% of the data')
        parts = name.split('_')
        enum = int(parts[2])
        tnum = int(parts[4].split('.')[0]) - 1
        # print(tnum)
        # print(enum)
        # print(data_list.keys())
        relevant_dict = data_list[-enum]['timestep_list'][tnum]
        # print(relevant_dict['reward'])
        # ADD IN CRITIC SO WE CAN SEE WHERE IT THINKS THE STATE IS GOOD
        
        state_labels = []
        state_data = []
        angs = [ang for ang in relevant_dict['state']['two_finger_gripper']['joint_angles'].values()]
        state_labels.append('Object Position')
        state_data.append(str(np.round(relevant_dict['state']['obj_2']['pose'][0], decimals=3)))
        state_labels.append('Object Orientation')
        state_data.append(str(np.round(relevant_dict['state']['obj_2']['pose'][1], decimals=3)))
        state_labels.append('Object Linear Velocity')
        state_data.append(str(np.round(relevant_dict['state']['obj_2']['velocity'][0],decimals=3)))
        state_labels.append('Object Angular')
        state_data.append(str(np.round(relevant_dict['state']['obj_2']['velocity'][1],decimals=3)))
        state_labels.append('Object Position')
        state_data.append(str(np.round(relevant_dict['state']['obj_2']['pose'][0], decimals=3)))
        state_labels.append('Object Orientation')
        state_data.append(str(np.round(relevant_dict['state']['obj_2']['pose'][1],decimals=3)))
        state_labels.append('Finger Angles')
        state_data.append(str(np.round(angs,decimals=3)))
        state_labels.append('F1 Position')
        state_data.append(str(np.round(relevant_dict['state']['f1_pos'],decimals=3)))
        state_labels.append('F2 Position')
        state_data.append(str(np.round(relevant_dict['state']['f2_pos'],decimals=3)))
        state_labels.append('F1, F2 finger distance')
        state_data.append(str(np.round([relevant_dict['state']['f1_obj_dist'],relevant_dict['state']['f2_obj_dist']],decimals=3)))
        a = [' '.join([i,j]) for i,j in zip(state_labels,state_data)]
        image = cv2.imread(image_path + name)
        actions = ' '.join(['Target Angle:',str(np.round(relevant_dict['action']['target_joint_angles'],decimals=2))])
        a2 = ' '.join(['Actor Output:', str(np.round(relevant_dict['control']['actor_output'],decimals=2))])
        a3 = ' '.join(['Critic Output:', str(np.round(relevant_dict['control']['critic_output'],decimals=3))])
        rewards = []
        rewards.append(['Goal Position', str(np.round(relevant_dict['reward']['goal_position'],decimals=3))])
        rewards.append(['Distance to goal', str(np.round(relevant_dict['reward']['distance_to_goal'],decimals=3))])
        rewards.append(['Reward F1 Finger Dist', str(np.round(relevant_dict['reward']['f1_dist'],decimals=3))])
        rewards.append(['Reward F2 Finger Dist', str(np.round(relevant_dict['reward']['f2_dist'],decimals=3))])
        rew = [' '.join([i[0],i[1]]) for i in rewards]
        y0, dy = 350,12
        cv2.putText(image,'Episode '+parts[2], (25, 330), font, 0.4, 2)
        for i, line in enumerate(a):
            y = y0 + i*dy
            cv2.putText(image, line, (25, y), font, 0.4, 2)
        cv2.putText(image, actions, (370, 350), font, 0.4, 2)
        cv2.putText(image, a2, (370, 362), font, 0.4, 2)
        cv2.putText(image, a3, (370, 374), font, 0.4, 2)
        y0, dy = 398,12
        for i, line in enumerate(rew):
            y = y0 + i*dy
            cv2.putText(image, line, (370, y), font, 0.4, 2)            
        cv2.imwrite('./labeled_images/'+name, image)
    print('processed 100% of the data')
        

if __name__ == "__main__":
    label_images('/home/orochi/mojo/mojo-grasp/demos/rl_demo/data/vizualization/')