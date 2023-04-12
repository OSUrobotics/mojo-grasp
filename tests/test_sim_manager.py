#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 09:46:10 2023

@author: orochi
"""

import unittest
import os
import json

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

class TestSimManager(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        print('setting up the class')
        with open('./test_configs/experiment_config.json','r') as configfile:
            args = json.load(configfile)
            
        x = [0.03, 0, -0.03, -0.04, -0.03, 0, 0.03, 0.04]
        y = [-0.03, -0.04, -0.03, 0, 0.03, 0.04, 0.03, 0]
            
        pose_list = [[i,j] for i,j in zip(x,y)]
        
        physics_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        p.resetDebugVisualizerCamera(cameraDistance=.02, cameraYaw=0, cameraPitch=-89.9999,
                                     cameraTargetPosition=[0, 0.1, 0.5])
        
        # load objects into pybullet
        plane_id = p.loadURDF("plane.urdf")
        hand_id = p.loadURDF(args['hand_path'], useFixedBase=True,
                             basePosition=[0.0, 0.0, 0.05])
        obj_id = p.loadURDF(args['object_path'], basePosition=[0.0, 0.16, .05])
        
        # Create TwoFingerGripper Object and set the initial joint positions
        hand = TwoFingerGripper(hand_id, path=args['hand_path'])
        
        p.resetJointState(hand_id, 0, .75)
        p.resetJointState(hand_id, 1, -1.4)
        p.resetJointState(hand_id, 3, -.75)
        p.resetJointState(hand_id, 4, 1.4)
        
        # change visual of gripper
        p.changeVisualShape(hand_id, 0, rgbaColor=[0.3, 0.3, 0.3, 1])
        p.changeVisualShape(hand_id, 1, rgbaColor=[1, 0.5, 0, 1])
        p.changeVisualShape(hand_id, 3, rgbaColor=[0.3, 0.3, 0.3, 1])
        p.changeVisualShape(hand_id, 4, rgbaColor=[1, 0.5, 0, 1])
        p.changeVisualShape(hand_id, -1, rgbaColor=[0.3, 0.3, 0.3, 1])
        # p.setTimeStep(1/2400)
        
        obj = ObjectWithVelocity(obj_id, path=args['object_path'])
        
        goal_poses = GoalHolder(pose_list)

        # state, action and reward
        state = StateRL(objects=[hand, obj, goal_poses], prev_len=args['pv'])
        action = rl_action.ExpertAction()
        reward = rl_reward.ExpertReward()
        
        #argument preprocessing
        arg_dict = args.copy()
        if args['action'] == 'Joint Velocity':
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
            data_path=args['save_path'], state=state, action=action, reward=reward, save_all=False, controller=manipulation.controller)
        
        # sim manager
        manager = SimManagerRLHER(num_episodes=len(pose_list), env=env, episode=EpisodeDefault(), record_data=record_data, replay_buffer=replay_buffer, state=state, action=action, reward=reward, args=arg_dict)
        
        # add phase to sim managerW
        manager.add_phase("manipulation", manipulation, start=True)
        
        # Run the sim
        for k in range(10):
            manager.run(test_flag=True)
            manager.phase_manager.phase_dict['manipulation'].reset()
        manager.save_network(args['save_path']+'policy')
        # replay_buffer.save_sampling('/home/orochi/mojo/mojo-grasp/demos/rl_demo/data/hard_random_sampling/sampling')
        print('training done, creating episode_all')
        d = data_processor(args['save_path'])
        if os.path.exists(args['save_path']+'episode_all.pkl'):
            os.remove(args['save_path']+'episode_all.pkl')
        d.load_data()
        d.save_all()
        cls._replay = replay_buffer
        cls.arg_dict = args
    
    def setUp(self):
        self.args = self.__class__.arg_dict
        self.replay_buffer = self.__class__._replay
        with open(self.args['save_path']+'episode_all.pkl', 'rb') as datafile:
            self.data = pkl.load(datafile)
        
    def testReplaySavedData(self):
        print('testing that replay buffer contains correct episodes')
        episode_counter = 0
        for i, episode in enumerate(self.data['episode_list']):
            for j, timestep in enumerate(episode['timestep_list']):
                # print(episode_counter+j)
                self.assertEqual(timestep['state'], self.replay_buffer.buffer_memory[episode_counter + j][0],'states in replay buffer dont match states in episode all'+str(i)+' '+str(j))
                self.assertEqual(timestep['action'], self.replay_buffer.buffer_memory[episode_counter + j][1],'actions in replay buffer dont match actions in episode all')
                self.assertEqual(timestep['reward'], self.replay_buffer.buffer_memory[episode_counter + j][2],'rewards in replay buffer dont match rewards in episode all')
                if j != 0:
                    self.assertEqual(timestep['state'], self.replay_buffer.buffer_memory[episode_counter + j-1][3],'next states in replay buffer dont match states in episode all')
                self.assertEqual(i+1,self.replay_buffer.buffer_memory[episode_counter + j][4],'replay buffer episode number incorrect')
            episode_counter += len(episode['timestep_list'])
        
    
    def testEpochOrder(self):
        print('testing that epochs are numbered and ordered correctly')
        print(self.replay_buffer.buffer_memory[0])
        self.assertEqual(len(self.replay_buffer), 80*151, 'replay buffer does not contain correct number of points')
        total_steps = [len(d['timestep_list']) for d in self.data['episode_list']]
        goal_poses = [d['timestep_list'][0]['state']['goal_pose']['goal_pose'] for d in self.data['episode_list']]
        self.assertEqual(sum(total_steps), 80*151, 'saved data does not contain correct number of points')
        count_contents, count_nums = np.unique(goal_poses, axis=0, return_counts=True)
        self.assertListEqual(count_nums.tolist(), [10]*8, 'goal poses sampled unevenly')
        x = [0.03, 0, -0.03, -0.04, -0.03, 0, 0.03, 0.04]
        y = [-0.03, -0.04, -0.03, 0, 0.03, 0.04, 0.03, 0]
        
        pose_list = [[i,j] for i,j in zip(x,y)]
        pose_len = len(pose_list)
        for i,pose in enumerate(goal_poses):
            self.assertListEqual(pose, pose_list[i%pose_len], 'goal pose doesnt match')
        # self.assertListEqual(pose_list, count_contents.tolist(),'goal poses dont match provided goal poses')
        
    def testReplayEpisode(self):
        print('testing that replay function results in same episode')
        x = [0.03, 0, -0.03, -0.04, -0.03, 0, 0.03, 0.04]
        y = [-0.03, -0.04, -0.03, 0, 0.03, 0.04, 0.03, 0]
            
        pose_list = [[i,j] for i,j in zip(x,y)]
        
        physics_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        p.resetDebugVisualizerCamera(cameraDistance=.02, cameraYaw=0, cameraPitch=-89.9999,
                                     cameraTargetPosition=[0, 0.1, 0.5])
        
        # load objects into pybullet
        plane_id = p.loadURDF("plane.urdf")
        hand_id = p.loadURDF(self.args['hand_path'], useFixedBase=True,
                             basePosition=[0.0, 0.0, 0.05])
        obj_id = p.loadURDF(self.args['object_path'], basePosition=[0.0, 0.16, .05])
        
        # Create TwoFingerGripper Object and set the initial joint positions
        hand = TwoFingerGripper(hand_id, path=self.args['hand_path'])
        
        p.resetJointState(hand_id, 0, .75)
        p.resetJointState(hand_id, 1, -1.4)
        p.resetJointState(hand_id, 3, -.75)
        p.resetJointState(hand_id, 4, 1.4)
        
        # change visual of gripper
        p.changeVisualShape(hand_id, 0, rgbaColor=[0.3, 0.3, 0.3, 1])
        p.changeVisualShape(hand_id, 1, rgbaColor=[1, 0.5, 0, 1])
        p.changeVisualShape(hand_id, 3, rgbaColor=[0.3, 0.3, 0.3, 1])
        p.changeVisualShape(hand_id, 4, rgbaColor=[1, 0.5, 0, 1])
        p.changeVisualShape(hand_id, -1, rgbaColor=[0.3, 0.3, 0.3, 1])
        # p.setTimeStep(1/2400)
        
        obj = ObjectWithVelocity(obj_id, path=self.args['object_path'])
        
        goal_poses = GoalHolder(pose_list)

        # state, action and reward
        state = StateRL(objects=[hand, obj, goal_poses], prev_len=self.args['pv'])
        action = rl_action.ExpertAction()
        reward = rl_reward.ExpertReward()
        
        #argument preprocessing
        arg_dict = self.args.copy()
        if self.args['action'] == 'Joint Velocity':
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
            data_path='./integration_test/', state=state, action=action, reward=reward, save_all=False, controller=manipulation.controller)
        
        # sim manager
        manager = SimManagerRLHER(num_episodes=len(pose_list), env=env, episode=EpisodeDefault(), record_data=record_data, replay_buffer=replay_buffer, state=state, action=action, reward=reward, args=arg_dict)
        
        # add phase to sim managerW
        manager.add_phase("manipulation", manipulation, start=True)
        manager.replay()
        
if __name__ == '__main__':
    unittest.main()