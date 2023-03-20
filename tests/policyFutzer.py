#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 12:31:06 2023

@author: orochi
"""
from multiprocessing import connection
from operator import truediv
import pybullet as p
import pybullet_data
import pathlib
from demos.rl_demo import rl_env
from demos.rl_demo import manipulation_phase_rl
# import rl_env
from demos.rl_demo.rl_state import StateRL, GoalHolder
from demos.rl_demo import rl_action
from demos.rl_demo import rl_reward
import pandas as pd
from mojograsp.simcore.sim_manager_HER import SimManagerRLHER
from mojograsp.simcore.state import StateDefault
from mojograsp.simcore.reward import RewardDefault
from mojograsp.simcore.record_data import RecordDataJSON, RecordDataPKL,  RecordDataRLPKL
from mojograsp.simobjects.two_finger_gripper import TwoFingerGripper
from mojograsp.simobjects.object_base import ObjectBase
from mojograsp.simobjects.object_with_velocity import ObjectWithVelocity
from mojograsp.simobjects.object_for_dataframe import ObjectVelocityDF
from mojograsp.simcore.replay_buffer import ReplayBufferDefault, ReplayBufferDF
from mojograsp.simcore.episode import EpisodeDefault
import numpy as np
from mojograsp.simcore.priority_replay_buffer import ReplayBufferPriority
from mojograsp.simcore.data_combination import data_processor
import pickle as pkl
from mojograsp.simcore.DDPGfD import DDPGfD_priority
import matplotlib.pyplot as plt
import json

class policy_Futzer():
    def __init__(self, filepath):
        
        with open(filepath+'experiment_config.json', 'r') as conf:
            args = json.load(conf)
        if args['action'] == 'Joint Velocity':
            ik_flag = False
        else:
            ik_flag = True
        arg_dict = {'state_dim': args['state_dim'], 'action_dim': 4, 'max_action': args['max_action'],
                    'n': 5, 'discount': args['discount'], 'tau': 0.0005,'batch_size': args['batch_size'], 
                    'epsilon':args['epsilon'], 'edecay': args['edecay'], 'ik_flag': ik_flag,
                    'reward':args['reward'], 'model':args['model'], 'tname':'no'}
        self.policy = DDPGfD_priority(args)
        self.policy.load(filepath+'policy')
        self.filepath = filepath
    
    def draw_critic_gradient(self, episode):
        with open(self.filepath + 'episode_' + episode + '.pkl','rb') as datafile:
            data = pkl.load(datafile)
        data = data['timestep_list']
        critic_vals, critic_grads = [],[]
        reward = []
        for timestep in data:
            t_state = timestep['state']
            state = []
            state.extend(t_state['obj_2']['pose'][0][0:2])
            state.extend(t_state['f1_pos'][0:2])
            state.extend(t_state['f2_pos'][0:2])               
            state.extend(timestep['reward']['goal_position'][0:2])
            action = timestep['action']['actor_output']
            cval, cgrad = self.policy.grade_action(t_state, action)
            critic_vals.append(cval)
            critic_grads.append(cgrad)
            tstep_reward = max(-timestep['reward']['distance_to_goal'] \
                - max(timestep['reward']['f1_dist'],timestep['reward']['f2_dist'])/5,-1)

            reward.append(tstep_reward)
        
        critic_grads = np.array(critic_grads)
        print(critic_grads)
        timesteps = list(range(len(reward)))
        
        plt.subplot(2, 1, 1)
        plt.title('reward and action gradient for episode ' + episode)
        plt.plot(timesteps, critic_vals)
        plt.plot(timesteps, reward)
        plt.xlabel('timesteps')
        plt.ylabel('state value')
        plt.legend(['Q value', 'Reward'])
        
        plt.subplot(2, 1, 2)
        plt.plot(timesteps,critic_grads[:,0])
        # plt.plot(timesteps,critic_grads[:,1])
        # plt.plot(timesteps,critic_grads[:,2])
        # plt.plot(timesteps,critic_grads[:,3])
        
        plt.xlabel('timesteps')
        plt.ylabel('action gradient')
        plt.legend(['F1 X', 'F1 Y', 'F2 X', 'F2 Y'])
        
        plt.show()


def main():


    aaaa = policy_Futzer('/home/orochi/mojo/mojo-grasp/demos/rl_demo/data/new_rollout/')
    aaaa.draw_critic_gradient('5000')


if __name__ == '__main__':
    main()