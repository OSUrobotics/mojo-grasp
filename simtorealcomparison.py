#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 11:28:50 2023

@author: orochi
"""
import json

import pickle as pkl


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

simpath = '/home/orochi/mojo/mojo-grasp/demos/rl_demo/data/hand_a_finger_pos_fulldimensionless/'
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