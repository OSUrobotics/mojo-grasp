#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 09:26:01 2023

@author: orochi
"""

import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
from mojograsp.simcore.data_gui_backend import PlotBackend

path = '/home/mothra/mojo-grasp/demos/rl_demo/data/'

folders = ['FTP_halfstate_A_rand','FTP_fullstate_A_rand','FTP_state_3_old',
           'JA_halfstate_A_rand','JA_fullstate_A_rand','JA_state_3_old']

plot_args = [['blue','solid'],
             ['orange','solid'],
             ['green','solid'],
             ['blue','dashed'],
             ['orange','dashed'],
             ['green','dashed']]

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
# for i,args in enumerate(plot_args):
#     plt.plot(range(10), range(i*10,i*10+10), color = args[0], linestyle=args[1])
# plt.show()
    
backend =PlotBackend(path+folders[0])
fig,ax = backend.get_figure()
backend.moving_avg = 1000
for folder,plot_stuff in zip(folders,plot_args):
    backend.draw_net_reward(path+folder+'/Test/',plot_args=plot_stuff)
plt.show()




