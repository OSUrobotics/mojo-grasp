#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 09:28:01 2023

@author: orochi
"""

import matplotlib.pyplot as plt
import imageio
import os
import numpy as np
import pickle as pkl

def load_pkl(filename):
    with open(filename, 'rb') as pkl_file:
        data_dict = pkl.load(pkl_file)
    data = data_dict['timestep_list']
    goal_position = data[0]['state']['goal_pose']['goal_pose']
    goal_dists = [f['reward']['distance_to_goal'] for f in data]
    ending_dist = min(goal_dists)
    return goal_position, ending_dist

filepath = '/home/orochi/mojo/mojo-grasp/demos/rl_demo/data/ftp_experiment/Train'
thold = 0.005

filenames = os.listdir(filepath)
episode_number = [int(filenam.split('_')[-1].split('.')[0]) for filenam in filenames]
# einds = np.argsort(episode_number)
max_num = max(episode_number)
successful_goals, failed_goals = [],[]
for i,episode in enumerate(episode_number):
    if episode/max_num > 0.99:
        goal, end_dist = load_pkl(filepath + '/'+filenames[i])
        if end_dist < thold:
            successful_goals.append(goal)
        else:
            failed_goals.append(goal)

successful_goals = np.array(successful_goals)
failed_goals = np.array(failed_goals)
plt.scatter(successful_goals[:,0], successful_goals[:,1])
plt.scatter(failed_goals[:,0], failed_goals[:,1])
plt.legend(['successful','failed'])
plt.title(f'Success Threshold: {thold*100} cm')
plt.show()