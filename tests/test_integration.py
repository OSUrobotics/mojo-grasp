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
import json



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
        with open('./test_configs/experiment_config.json','r') as configfile:
            self.arg_dict = json.load(configfile)
        self.arg_dict['action_dim'] = 4
        self.arg_dict['tau'] = 0.0005
        self.arg_dict['n'] = 5
        
        x = [0.01, 0.0]
        y = [0.0, 0.01]
        physics_client = p.connect(p.DIRECT)
        pose_list = [[i,j] for i,j in zip(x,y)]
        eval_pose_list = pose_list
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        p.resetDebugVisualizerCamera(cameraDistance=.02, cameraYaw=0, cameraPitch=-89.9999,
                                     cameraTargetPosition=[0, 0.1, 0.5])
        
        # load objects into pybullet
        plane_id = p.loadURDF("plane.urdf", flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
        hand_id = p.loadURDF(self.arg_dict['hand_path'], useFixedBase=True,
                             basePosition=[0.0, 0.0, 0.05], flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
        obj_id = p.loadURDF(self.arg_dict['object_path'], basePosition=[0.0, 0.10, .05], flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
        
        # Create TwoFingerGripper Object and set the initial joint positions
        hand = TwoFingerGripper(hand_id, path=self.arg_dict['hand_path'])
        
        # p.resetJointState(hand_id, 0, -0.4)
        # p.resetJointState(hand_id, 1, 1.2)
        # p.resetJointState(hand_id, 3, 0.4)
        # p.resetJointState(hand_id, 4, -1.2)
        
        # p.resetJointState(hand_id, 0, 0)
        # p.resetJointState(hand_id, 1, 0)
        # p.resetJointState(hand_id, 3, 0)
        # p.resetJointState(hand_id, 4, 0)
        # change visual of gripper
        p.changeVisualShape(hand_id, 0, rgbaColor=[0.3, 0.3, 0.3, 1])
        p.changeVisualShape(hand_id, 1, rgbaColor=[1, 0.5, 0, 1])
        p.changeVisualShape(hand_id, 3, rgbaColor=[0.3, 0.3, 0.3, 1])
        p.changeVisualShape(hand_id, 4, rgbaColor=[1, 0.5, 0, 1])
        p.changeVisualShape(hand_id, -1, rgbaColor=[0.3, 0.3, 0.3, 1])
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
        # p.setTimeStep(1/2400)
        obj = ObjectWithVelocity(obj_id, path=self.arg_dict['object_path'])
        # p.addUserDebugPoints([[0.2,0.1,0.0],[1,0,0]],[[1,0.0,0],[0.5,0.5,0.5]], 1)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        goal_poses = GoalHolder(pose_list)
        eval_goal_poses = GoalHolder(eval_pose_list)
        # time.sleep(10)
        # state, action and reward
        state = StateRL(objects=[hand, obj, goal_poses], prev_len=self.arg_dict['pv'],eval_goals = eval_goal_poses)
        action = rl_action.ExpertAction()
        reward = rl_reward.ExpertReward()
        
        #argument preprocessing
        arg_dict = self.arg_dict.copy()
        if self.arg_dict['action'] == 'Joint Velocity':
            arg_dict['ik_flag'] = False
        else:
            arg_dict['ik_flag'] = True
    
        # replay buffer
        replay_buffer = ReplayBufferPriority(buffer_size=4080000)
        
        # environment and recording
        env = rl_env.ExpertEnv(hand=hand, obj=obj)
        # env = rl_env.ExpertEnv(hand=hand, obj=cylinder)
        
        # Create phase
        manipulation = manipulation_phase_rl.ManipulationRL(
            hand, obj, x, y, state, action, reward, replay_buffer=replay_buffer, args=arg_dict)
    
        # data recording
        record_data = RecordDataRLPKL(
            data_path=self.arg_dict['save_path'], state=state, action=action, reward=reward, save_all=False, controller=manipulation.controller)
        
        # sim manager
        self.manager = SimManagerRLHER(num_episodes=len(pose_list), env=env, episode=EpisodeDefault(), record_data=record_data, replay_buffer=replay_buffer, state=state, action=action, reward=reward, args=arg_dict)
        
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