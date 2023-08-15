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

def load_pkl_goal_dist(filename,last=False):
    with open(filename, 'rb') as pkl_file:
        data_dict = pkl.load(pkl_file)
    data = data_dict['timestep_list']
    goal_position = data[0]['state']['goal_pose']['goal_pose']
    
    if last:
        ending_dist = data[-1]['reward']['distance_to_goal']
    else:
        goal_dists = [f['reward']['distance_to_goal'] for f in data]
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

def load_pkls_compare(filepath1, filepath2, thold):
    goal_poses1, end_dist1 = load_pkl_goal_dist(filepath1)
    goal_poses2, end_dist2 = load_pkl_goal_dist(filepath2)
    successful_1_goals = []
    successful_2_goals = []
    for goal1,end1 in zip(goal_poses1, end_dist1):
        if end1 < thold:
            successful_1_goals.append(goal1)
    for goal2,end2 in zip(goal_poses2, end_dist2):
        if end2 < thold:
            successful_2_goals.append(goal2)
            
    
    # return goal_position, end_position


# filepath = '/home/orochi/mojo/mojo-grasp/demos/rl_demo/data/ftp_comparison/EVALUATION/Test'
# filepath = '/home/orochi/mojo/mojo-grasp/demos/rl_demo/data/ja_testing/EVALUATION/Test'
# filepath = '/home/orochi/mojo/mojo-grasp/demos/rl_demo/data/ja_abinav_rewards/eval'
# filepath = '/home/orochi/mojo/mojo-grasp/demos/rl_demo/data/ja_abinav_rewards_higher_speed/eval_stuff'
filepath = '/home/orochi/mojo/mojo-grasp/demos/rl_demo/data/ja_new_rewards/eval'
# filepath = '/home/orochi/mojo/mojo-grasp/demos/rl_demo/data/IK_results/'

thold = 0.01


filenames = os.listdir(filepath)
print(len(filenames))
episode_number = [int(re.search('\d+',filenam)[0]) for filenam in filenames]
einds = np.argsort(episode_number)
new_name_order = [filenames[i] for i in einds]
# print(new_name_order)
max_num = max(episode_number)


# successful_goals, failed_goals = [],[]
# for i,episode in enumerate(episode_number):
#     # print(filenames[i])
#     goal, end_dist = load_pkl_goal_dist(filepath + '/'+filenames[i])
#     # print(end_dist)
#     if end_dist < thold:
#         successful_goals.append(goal)
#     else:
#         failed_goals.append(goal)

# # print(successful_goals)
# successful_goals = np.array(successful_goals)
# failed_goals = np.array(failed_goals)
# plt.scatter(successful_goals[:,0], successful_goals[:,1])
# plt.scatter(failed_goals[:,0], failed_goals[:,1])
# plt.legend(['successful','failed'])
# plt.title(f'Success Threshold: {thold*100} cm')


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

# maintain_contact, failed_contact = [],[]
# contact_dist = []
# for i,episode in enumerate(new_name_order):
#     goal, finger_dists, tsteps_with_contact = load_pkl_finger_dist(filepath + '/' + episode)
#     contact_dist.append(np.max(finger_dists))


# plt.plot(range(len(contact_dist)), contact_dist)
# plt.title(f'Contact Threshold: {thold*1000} mm')
# plt.xlabel('Epoch')
# plt.ylabel('Maximum Finger Tip Distance')

# nope, singularity = [],[]
# for i,episode in enumerate(episode_number):
#     goal, eval_s = load_pkl_angle(filepath + '/' + filenames[i])
#     if np.isclose(eval_s,0, atol=0.01).any():
#         singularity.append(goal)
#         print('file is singularity', filenames[i])
#     else:
#         nope.append(goal)

# success, fail = [],[]
# for i,episode in enumerate(episode_number):
#     goal, end_dist = load_pkl_goal_dist(filepath + '/' + filenames[i])
#     if end_dist < thold:
#         success.append(goal)
#     else:
#         fail.append(goal)

# singular_success= 0
# singular_fail = 0
# for goal in singularity:
#     if goal in success:
#         singular_success+=1
#     elif goal in fail:
#         singular_fail +=1

# norm_success= 0
# norm_fail = 0
# for goal in nope:
#     if goal in success:
#         norm_success+=1
#     elif goal in fail:
#         norm_fail +=1
        
# print(f'when we hit a singularity we succeed {singular_success/(singular_success+singular_fail)*100}% of the time')
# print(f'when we dont hit a singularity we succeed {norm_success/(norm_success+norm_fail)*100}% of the time')
# print(singular_success,singular_fail, norm_success,norm_fail)
# nope = np.array(nope)
# singularity = np.array(singularity)
# plt.scatter(nope[:,0], nope[:,1])
# plt.scatter(singularity[:,0], singularity[:,1])
# plt.legend(['NO singularity','singularity'])
# plt.title('Points that reach a singlularity (within 0.01 rads of straight finger)')


# angle_difference, singularity = [],[]
# for i,episode in enumerate(episode_number):
#     goal, angles = load_pkl_angle(filepath + '/' + filenames[i])
    
# nope = np.array(nope)
# singularity = np.array(singularity)
# plt.scatter(nope[:,0], nope[:,1])
# plt.scatter(singularity[:,0], singularity[:,1])
# plt.legend(['NO singularity','singularity'])
# plt.title('Fingers didnt get within 0.001 rads of singularity')