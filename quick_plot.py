#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 09:26:01 2023

@author: orochi
"""

import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np

path = '/home/orochi/mojo/mojo-grasp/demos/rl_demo/data/real_world/'

def draw_path(episode):
    data = episode['timestep_list']
    trajectory_points = [f['state']['current_state']['obj_2']['pose'][0] for f in data]
    goal_pose = data[1]['state']['current_state']['goal_pose']['goal_pose']
    trajectory_points = np.array(trajectory_points)

    plt.plot(trajectory_points[:,0], trajectory_points[:,1])
    plt.plot([trajectory_points[0,0], goal_pose[0]],[trajectory_points[0,1],goal_pose[1]+0.1])
    plt.xlim([-0.07,0.07])
    plt.ylim([0.04,0.16])
    plt.xlabel('X pos (m)')
    plt.ylabel('Y pos (m)')                                                                                                                                                                                                                                   
    legend = ['RL Trajectory','Ideal Path to Goal']
    # plt.legend(legend)
    plt.title('Object Path')

with open(path+'N_JA_episode.pkl','rb') as file:
    data = pkl.load(file)
draw_path(data)



with open(path+'E_JA_episode.pkl','rb') as file:
    data = pkl.load(file)
draw_path(data)


with open(path+'S_JA_episode.pkl','rb') as file:
    data = pkl.load(file)
draw_path(data)  

with open(path+'W_JA_episode.pkl','rb') as file:
    data = pkl.load(file)
draw_path(data)




with open(path+'SW_JA_episode.pkl','rb') as file:
    data = pkl.load(file)
draw_path(data)


with open(path+'NW_JA_episode.pkl','rb') as file:
    data = pkl.load(file)
draw_path(data)  
with open(path+'NE_JA_episode.pkl','rb') as file:
    data = pkl.load(file)
draw_path(data)  
with open(path+'SE_JA_episode.pkl','rb') as file:
    data = pkl.load(file)
draw_path(data)
plt.show()