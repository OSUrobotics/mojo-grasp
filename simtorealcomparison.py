#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 11:28:50 2023

@author: orochi
"""
import json

import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np


def build_state_sim(state_container, PREV_VALS, state_list):
    """
    Method takes in a State object 
    Extracts state information from state_container and returns it as a list based on
    current used states contained in self.state_list

    :param state: :func:`~mojograsp.simcore.phase.State` object.
    :type state: :func:`~mojograsp.simcore.phase.State`
    """
    state = []
    if PREV_VALS > 0:
        for i in range(PREV_VALS):
            for key in state_list:
                if key == 'op':
                    state.extend(state_container['previous_state'][i]['obj_2']['pose'][0][0:2])
                elif key == 'ftp':
                    state.extend(state_container['previous_state'][i]['f1_pos'][0:2])
                    state.extend(state_container['previous_state'][i]['f2_pos'][0:2])
                elif key == 'fbp':
                    state.extend(state_container['previous_state'][i]['f1_base'][0:2])
                    state.extend(state_container['previous_state'][i]['f2_base'][0:2])
                elif key == 'fcp':
                    state.extend(state_container['previous_state'][i]['f1_contact_pos'][0:2])
                    state.extend(state_container['previous_state'][i]['f2_contact_pos'][0:2])
                elif key == 'ja':
                    state.extend([item for item in state_container['previous_state'][i]['two_finger_gripper']['joint_angles'].values()])
                elif key == 'fta':
                    state.extend([state_container['previous_state'][i]['f1_ang'],state_container['previous_state'][i]['f2_ang']])
                elif key == 'gp':
                    state.extend(state_container['previous_state'][i]['goal_pose']['goal_pose'])
                else:
                    raise Exception('key does not match list of known keys')

    for key in state_list:
        if key == 'op':
            state.extend(state_container['obj_2']['pose'][0][0:2])
        elif key == 'ftp':
            state.extend(state_container['f1_pos'][0:2])
            state.extend(state_container['f2_pos'][0:2])
        elif key == 'fbp':
            state.extend(state_container['f1_base'][0:2])
            state.extend(state_container['f2_base'][0:2])
        elif key == 'fcp':
            state.extend(state_container['f1_contact_pos'][0:2])
            state.extend(state_container['f2_contact_pos'][0:2])
        elif key == 'ja':
            state.extend([item for item in state_container['two_finger_gripper']['joint_angles'].values()])
        elif key == 'fta':
            state.extend([state_container['f1_ang'],state_container['f2_ang']])
        elif key == 'gp':
            state.extend(state_container['goal_pose']['goal_pose'])
        else:
            raise Exception('key does not match list of known keys')
    return state
def build_state_real(state_container, PREV_VALS, state_list):
    """
    Method takes in a State object 
    Extracts state information from state_container and returns it as a list based on
    current used states contained in self.state_list

    :param state: :func:`~mojograsp.simcore.phase.State` object.
    :type state: :func:`~mojograsp.simcore.phase.State`
    """
    state = []
    if PREV_VALS > 0:
        for i in range(PREV_VALS):
            for key in state_list:
                if key == 'op':
                    state.extend(state_container['previous_state'][i]['obj_2']['pose'][0][0:2])
                elif key == 'ftp':
                    state.extend(state_container['previous_state'][i]['f1_pos'][0:2])
                    state.extend(state_container['previous_state'][i]['f2_pos'][0:2])
                elif key == 'fbp':
                    state.extend(state_container['previous_state'][i]['f1_base'][0:2])
                    state.extend(state_container['previous_state'][i]['f2_base'][0:2])
                elif key == 'fcp':
                    state.extend(state_container['previous_state'][i]['f1_contact_pos'][0:2])
                    state.extend(state_container['previous_state'][i]['f2_contact_pos'][0:2])
                elif key == 'ja':
                    state.extend([item for item in state_container['previous_state'][i]['two_finger_gripper']['joint_angles'].values()])
                elif key == 'fta':
                    state.extend([state_container['previous_state'][i]['f1_ang'],state_container['previous_state'][i]['f2_ang']])
                elif key == 'gp':
                    state.extend(state_container['previous_state'][i]['goal_pose']['goal_pose'])
                else:
                    raise Exception('key does not match list of known keys')

    for key in state_list:
        if key == 'op':
            state.extend(state_container['current_state']['obj_2']['pose'][0][0:2])
        elif key == 'ftp':
            state.extend(state_container['current_state']['f1_pos'][0:2])
            state.extend(state_container['current_state']['f2_pos'][0:2])
        elif key == 'fbp':
            state.extend(state_container['current_state']['f1_base'][0:2])
            state.extend(state_container['current_state']['f2_base'][0:2])
        elif key == 'fcp':
            state.extend(state_container['current_state']['f1_contact_pos'][0:2])
            state.extend(state_container['current_state']['f2_contact_pos'][0:2])
        elif key == 'ja':
            state.extend([item for item in state_container['current_state']['two_finger_gripper']['joint_angles'].values()])
        elif key == 'fta':
            state.extend([state_container['current_state']['f1_ang'],state_container['current_state']['f2_ang']])
        elif key == 'gp':
            state.extend(state_container['current_state']['goal_pose']['goal_pose'])
        else:
            raise Exception('key does not match list of known keys')
    return state

def build_state(state_container, state_list):
    """
    Method takes in a State object 
    Extracts state information from state_container and returns it as a list based on
    current used states contained in self.state_list

    :param state: :func:`~mojograsp.simcore.phase.State` object.
    :type state: :func:`~mojograsp.simcore.phase.State`
    """
    state = []
    for key in state_list:
        if key == 'op':
            state.extend(state_container['obj_2']['pose'][0][0:2])
        elif key == 'ftp':
            state.extend(state_container['f1_pos'][0:2])
            state.extend(state_container['f2_pos'][0:2])
        elif key == 'fbp':
            state.extend(state_container['f1_base'][0:2])
            state.extend(state_container['f2_base'][0:2])
        elif key == 'fcp':
            state.extend(state_container['f1_contact_pos'][0:2])
            state.extend(state_container['f2_contact_pos'][0:2])
        elif key == 'ja':
            state.extend([item for item in state_container['two_finger_gripper']['joint_angles'].values()])
        elif key == 'fta':
            state.extend([state_container['f1_ang'],state_container['f2_ang']])
        elif key == 'gp':
            state.extend(state_container['goal_pose']['goal_pose'])
        else:
            raise Exception('key does not match list of known keys')
    return state


        
def draw_parameter_comparison_multi(sim_dictionary,real_list,state_key):
    data = sim_dictionary['timestep_list']
    sim_state = []
    real_state = []
    
    temp_sim_state = [f['state'][state_key][0] for f in data]
    sim_state.append(temp_sim_state)
    temp_sim_state = [f['state'][state_key][1] for f in data]
    sim_state.append(temp_sim_state)
    temp_real_state = [f['current_state'][state_key][0] for f in real_list]
    real_state.append(temp_real_state)
    temp_real_state = [f['current_state'][state_key][1] for f in real_list]
    real_state.append(temp_real_state)
    #sim_state = np.array(sim_state)
    #real_state = np.array(real_state)
    legend = []
    
    plt.plot(range(len(sim_state[0])),sim_state[0])
    plt.plot(range(len(real_state[0])),real_state[0])
    plt.plot(range(len(sim_state[1])),sim_state[1])
    plt.plot(range(len(real_state[1])),real_state[1])
    legend.extend(['Sim ' + state_key + ' x', 'Real ' + state_key+ ' x','Sim ' + state_key + ' y', 'Real ' + state_key+ ' y'])
    plt.legend(legend)
    
    plt.grid(True)
    plt.ylabel('State Comparison')
    plt.xlabel('Timestep (1/30 s)')
    
def draw_parameter_comparison(sim_dictionary,real_list,state_key_list):
    data = sim_dictionary['timestep_list']
    sim_state = []
    real_state = []
    for key in state_key_list:
        temp_sim_state = [f['state'][key] for f in data]
        sim_state.append(temp_sim_state)
        temp_real_state = [f['state']['current_state'][key] for f in real_list['timestep_list']]
        real_state.append(temp_real_state)
    #sim_state = np.array(sim_state)
    #real_state = np.array(real_state)
    legend = []
    for i,key in enumerate(state_key_list):
        plt.plot(range(len(sim_state[i])),sim_state[i])
        plt.plot(range(len(real_state[i])),real_state[i])
        legend.extend(['Sim ' + key, 'Real ' + key])
    plt.legend(legend)
    
    plt.grid(True)
    plt.ylabel('State Comparison')
    plt.xlabel('Timestep (1/30 s)')
    
def draw_path(sim_dict, real_dict):
    simdata = sim_dict['timestep_list']
    realdata = real_dict
    simtrajectory_points = [f['state']['obj_2']['pose'][0] for f in simdata]
    realtrajectory_points = [f['current_state']['obj_2']['pose'][0] for f in realdata]
    goal_pose = simdata[1]['reward']['goal_position']
    simtrajectory_points = np.array(simtrajectory_points)
    realtrajectory_points = np.array(realtrajectory_points)
    plt.plot(simtrajectory_points[:,0], simtrajectory_points[:,1])
    plt.plot([simtrajectory_points[0,0], goal_pose[0]],[simtrajectory_points[0,1],goal_pose[1]])
    plt.plot(realtrajectory_points[:,0],realtrajectory_points[:,1])
    plt.xlim([-0.07,0.07])
    plt.ylim([0.04,0.16])
    plt.xlabel('X pos (m)')
    plt.ylabel('Y pos (m)')                                                                                                                                                                                                                                   
    plt.legend(['Sim RL Trajectory', 'Ideal Path to Goal', 'Real RL Trajectory'])
    plt.title('Object Path')
    
simpath = '/home/orochi/mojo/mojo-grasp/demos/rl_demo/data/PPO_JA_long/'
realpath = '/home/orochi/mojo/mojo-grasp/demos/rl_demo/data/real_world/'

with open(simpath+'experiment_config.json','r') as File:
    args = json.load(File)

with open(simpath+'Train/episode_5000.pkl','rb') as File:
    simdata = pkl.load(File)


with open(realpath+'E_episode.pkl','rb') as File:
    realdata = pkl.load(File)


sim_built_state = build_state(simdata['timestep_list'][0]['state'], args['state_list'])
real_built_state = build_state(realdata['timestep_list'][0]['state']['current_state'], args['state_list'])
sim_built_state[-2:] = real_built_state[-2:]
errors = []
for i in range(len(sim_built_state)):
    value_range = args['state_maxes'][i] - args['state_mins'][i]
    percent_error = abs(sim_built_state[i] - real_built_state[i])/value_range
    errors.append(percent_error)
    if percent_error >0.03:
        print('ya fucked up')
        print(sim_built_state[i], real_built_state[i])