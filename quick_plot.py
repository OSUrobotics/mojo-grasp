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


folders = ['JA_newstate_A_rand','JA_fullstate_A_rand','JA_halfstate_A_rand',  'FTP_newstate_A_rand', 'FTP_fullstate_A_rand', 'FTP_halfstate_A_rand']
PB = PlotBackend(path+ folders[0])
PB.clear_plots = False
PB.moving_avg = 1000
figures, _ = PB.get_figure()
for folder in folders:
    PB.draw_net_reward(path+folder+'/Test')

plt.show()