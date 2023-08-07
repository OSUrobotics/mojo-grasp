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
import re

def load_pkl_goal_dist(filename,mins=False):
    with open(filename, 'rb') as pkl_file:
        data_dict = pkl.load(pkl_file)
    data = data_dict['timestep_list']
    goal_position = data[0]['state']['goal_pose']['goal_pose']
    goal_dists = [f['reward']['distance_to_goal'] for f in data]
    if mins:
        ending_dist = goal_dists[-1]
    else:
        ending_dist = min(goal_dists)

    return goal_position, ending_dist

def load_pkl_end_pos(filename):
    with open(filename, 'rb') as pkl_file:
        data_dict = pkl.load(pkl_file)
    data = data_dict['timestep_list']
    end_position = data[-1]['state']['obj_2']['pose'][0]
    goal_position = data[0]['state']['goal_pose']['goal_pose'][0:2]
    return goal_position, end_position

def load_pkl_finger_dist(filename):
    with open(filename, 'rb') as pkl_file:
        data_dict = pkl.load(pkl_file)
    data = data_dict['timestep_list']
    f1_dists = [f['reward']['f1_dist'] for f in data]
    f2_dists = [f['reward']['f2_dist'] for f in data]
    goal_position = data[0]['state']['goal_pose']['goal_pose'][0:2]
    tsteps_with_contact = []
    for f1,f2,i in zip(f1_dists,f2_dists,range(151)):
        if max([f1,f2]) < 0.001:
            tsteps_with_contact.append(i)
    return goal_position, np.array([f1_dists, f2_dists]), tsteps_with_contact

def load_pkl_eigen(filename):
    with open(filename, 'rb') as pkl_file:
        data_dict = pkl.load(pkl_file)
    data = data_dict['timestep_list']
    eigenvectors = [f['state']['two_finger_gripper']['eigenvectors'] for f in data]
    eigenvalues = [f['state']['two_finger_gripper']['eigenvalues'] for f in data]
    goal_position = data[0]['state']['goal_pose']['goal_pose'][0:2]

    return goal_position, eigenvalues

def load_pkl_angle(filename):
    with open(filename, 'rb') as pkl_file:
        data_dict = pkl.load(pkl_file)
    data = data_dict['timestep_list']
    current_angle = [[f['state']['two_finger_gripper']['joint_angles']['finger0_segment1_joint'],
                           f['state']['two_finger_gripper']['joint_angles']['finger1_segment1_joint']] for f in data]
    
    goal_position = data[0]['state']['goal_pose']['goal_pose'][0:2]

    return goal_position, current_angle


filepath = '/home/orochi/mojo/mojo-grasp/demos/rl_demo/data/ja_testing/EVALUATION/Test'
thold = 0.001

filenames = os.listdir(filepath)
episode_number = [re.search('\d+',filenam)[0] for filenam in filenames]
# einds = np.argsort(episode_number)
# max_num = max(episode_number)
# successful_goals, failed_goals = [],[]
# for i,episode in enumerate(episode_number):
#     goal, end_dist = load_pkl(filepath + '/'+filenames[i])
#     if end_dist < thold:
#         successful_goals.append(goal)
#     else:
#         failed_goals.append(goal)

# successful_goals = np.array(successful_goals)
# failed_goals = np.array(failed_goals)
# plt.scatter(successful_goals[:,0], successful_goals[:,1])
# plt.scatter(failed_goals[:,0], failed_goals[:,1])
# plt.legend(['successful','failed'])
# plt.title(f'Success Threshold: {thold*100} cm')

'''
successful_goals, failed_goals = [],[]
for i,episode in enumerate(episode_number):
    goal, end_dist = load_pkl_goal_dist(filepath + '/'+filenames[i], True)
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
'''

# goals, end_spots = [],[]
# for i,episode in enumerate(episode_number):
#     goal, end_dist = load_pkl_end_pos(filepath + '/' + filenames[i])
#     goals.append(goal)
#     end_spots.append(end_dist)

# successful_goals = np.array(goals)
# failed_goals = np.array(end_spots) - np.array([0,0.1,0])
# plt.scatter(successful_goals[:,0], successful_goals[:,1])
# plt.scatter(failed_goals[:,0], failed_goals[:,1])
# plt.legend(['Goal Poses','End Poses'])
# plt.title('Goal and End Poses')

# maintain_contact, failed_contact = [],[]
# for i,episode in enumerate(episode_number):
#     goal, finger_dists, tsteps_with_contact = load_pkl_finger_dist(filepath + '/' + filenames[i])
    
#     if np.max(finger_dists) < thold:
#         maintain_contact.append(goal)
#     else:
#         failed_contact.append(goal)
# maintain_contact = np.array(maintain_contact)
# failed_contact = np.array(failed_contact)
# plt.scatter(maintain_contact[:,0], maintain_contact[:,1])
# plt.scatter(failed_contact[:,0], failed_contact[:,1])
# plt.legend(['successful','failed'])
# plt.title(f'Contact Threshold: {thold*1000} mm')



nope, singularity = [],[]
for i,episode in enumerate(episode_number):
    goal, eval_s = load_pkl_angle(filepath + '/' + filenames[i])
    if np.isclose(eval_s,0, atol=0.0001).any():
        singularity.append(goal)
    else:
        nope.append(goal)
nope = np.array(nope)
singularity = np.array(singularity)
plt.scatter(nope[:,0], nope[:,1])
plt.scatter(singularity[:,0], singularity[:,1])
plt.legend(['NO singularity','singularity'])
plt.title('Fingers didnt get within 0.0001 rads of singularity')