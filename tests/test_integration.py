#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 10:29:26 2023

@author: orochi
"""


import unittest
from mojograsp.simcore.DDPGfD import DDPGfD_priority
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




def calc_finger_poses(angles):
    x0 = [-0.02675, 0.02675]
    y0 = [0.053, 0.053]
    f1x = x0[0] - np.sin(angles[0])*0.072 - np.sin(angles[0] + angles[1])*0.072
    f2x = x0[1] - np.sin(angles[2])*0.072 - np.sin(angles[2] + angles[3])*0.072
    f1y = y0[0] + np.cos(angles[0])*0.072 + np.cos(angles[0] + angles[1])*0.072
    f2y = y0[1] + np.cos(angles[2])*0.072 + np.cos(angles[2] + angles[3])*0.072
    
    return [f1x, f1y, f2x, f2y]

class fack():
    def __init__(self):
        with open('./test_configs/experiment_config.pkl','rb') as configfile:
            self.arg_dict = pkl.load(configfile)
        self.arg_dict['action_dim'] = 4
        self.arg_dict['tau'] = 0.0005
        self.arg_dict['n'] = 5
        
        x = [0.01, 0.0]
        y = [0.0, 0.01]
        pose_list = [[i,j] for i,j in zip(x,y)]
        
    
        # physics_client = p.connect(p.GUI)
        physics_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        p.resetDebugVisualizerCamera(cameraDistance=.02, cameraYaw=0, cameraPitch=-89.9999,
                                     cameraTargetPosition=[0, 0.1, 0.5])
        
        # load objects into pybullet
        plane_id = p.loadURDF("plane.urdf")
        hand_id = p.loadURDF(self.arg_dict['hand_path'], useFixedBase=True,
                             basePosition=[0.0, 0.0, 0.05])
        obj_id = p.loadURDF(self.arg_dict['object_path'], basePosition=[0.0, 0.16, .05])
        
        # Create TwoFingerGripper Object and set the initial joint positions
        hand = TwoFingerGripper(hand_id, path=self.arg_dict['hand_path'])
        
        p.resetJointState(hand_id, 0, .75)
        p.resetJointState(hand_id, 1, -1.4)
        p.resetJointState(hand_id, 3, -.75)
        p.resetJointState(hand_id, 4, 1.4)

        # p.resetJointState(hand_id, 0, 0)
        # p.resetJointState(hand_id, 1, 0)
        # p.resetJointState(hand_id, 3, 0)
        # p.resetJointState(hand_id, 4, 0)

        
        # p.resetJointState(hand_id, 0, .695)
        # p.resetJointState(hand_id, 1, -1.487)
        # p.resetJointState(hand_id, 3, -.695)
        # p.resetJointState(hand_id, 4, 1.487)
        
        # cylinder = ObjectWithVelocity(cylinder_id, path=cylinder_path)
        # obj = ObjectVelocityDF(obj_id, path=self.arg_dict['object_path'])
        
        
        # change visual of gripper
        p.changeVisualShape(hand_id, 0, rgbaColor=[0.3, 0.3, 0.3, 1])
        p.changeVisualShape(hand_id, 1, rgbaColor=[1, 0.5, 0, 1])
        p.changeVisualShape(hand_id, 3, rgbaColor=[0.3, 0.3, 0.3, 1])
        p.changeVisualShape(hand_id, 4, rgbaColor=[1, 0.5, 0, 1])
        p.changeVisualShape(hand_id, -1, rgbaColor=[0.3, 0.3, 0.3, 1])
        # p.setTimeStep(1/2400)
        
        obj = ObjectWithVelocity(obj_id, path=self.arg_dict['object_path'])
        
        goal_poses = GoalHolder(pose_list)
        # state and reward
        # state = StateDefault(objects=[hand, obj])
        state = StateRL(objects=[hand, obj, goal_poses])
        # state = StateRL(objects=[hand, cylinder, goal_poses])
        action = rl_action.ExpertAction()
        reward = rl_reward.ExpertReward()
        if self.arg_dict['action'] == 'Joint Velocity':
            ik_flag = False
        else:
            ik_flag = True
        fake_arg_dict = {'state_dim': self.arg_dict['state_dim'], 'action_dim': 4, 'max_action': self.arg_dict['max_action'],
                    'n': 5, 'discount': self.arg_dict['discount'], 'tau': 0.0005,'batch_size': 100000000, 
                    'epsilon':self.arg_dict['epsilon'], 'edecay': self.arg_dict['edecay'], 'ik_flag': True,
                    'reward':self.arg_dict['reward'], 'model':self.arg_dict['model']}
        # arg_dict = {'state_dim': 8, 'action_dim': 4, 'max_action': 0.005, 'n': 5, 'discount': 0.995, 'tau': 0.0005,
        #             'batch_size': 100, 'expert_sampling_proportion': 0.7}
        
        # replay buffer
        replay_buffer = ReplayBufferPriority(buffer_size=4080000)
        # replay_buffer.preload_buffer_PKL(self.arg_dict['edata'])
        # replay_buffer = ReplayBufferDF(state=state, action=action, reward=reward)
        
        
        # environment and recording
        env = rl_env.ExpertEnv(hand=hand, obj=obj)
        # env = rl_env.ExpertEnv(hand=hand, obj=cylinder)
        
        # Create phase
        manipulation = manipulation_phase_rl.ManipulationRL(
            hand, obj, x, y, state, action, reward, replay_buffer=replay_buffer, args=fake_arg_dict, tbname=self.arg_dict['save_path'])
        # manipulation = manipulation_phase_rl.ManipulationRL(
        #     hand, cylinder, x, y, state, action, reward, replay_buffer=replay_buffer, self.arg_dict=arg_dict)
        # data recording
        record_data = RecordDataRLPKL(
            data_path=self.arg_dict['save_path'], state=state, action=action, reward=reward, save_all=False, controller=manipulation.controller)
        
        # sim manager
        self.manager = SimManagerRLHER(num_episodes=len(pose_list), env=env, episode=EpisodeDefault(), record_data=record_data, replay_buffer=replay_buffer, state=state, action=action, reward=reward, TensorboardName='test1', args=fake_arg_dict)
        
        # add phase to sim manager
        self.manager.add_phase("manipulation", manipulation, start=True)
        
        # load up replay buffer
        # for i in range(4):
        #     manager.run()
        #     manager.phase_manager.phase_dict['manipulation'].reset()
        #print(p.getClosestPoints(obj.id, hand.id, 1, -1, 1, -1))
        # Run the 
        
        
    def test_sim_manager(self):
        self.manager.run()
        self.manager.phase_manager.phase_dict['manipulation'].reset()
        
    def test_recorded_data(self):
        # this is gonna be gross
        replay_buffer = self.manager.replay_buffer
        recorded_data = self.manager.record
    
        
if __name__ == '__main__':
    a = fack()
    a.test_sim_manager()