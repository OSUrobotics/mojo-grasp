#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 13:11:51 2023

@author: orochi
"""
from multiprocessing import connection
from operator import truediv
import pybullet as p
import pybullet_data
import pathlib
import manipulation_phase_rl
import rl_env
from rl_state import StateRL, GoalHolder
import rl_action
import rl_reward
import pandas as pd
from mojograsp.simcore.sim_manager import SimManagerRL
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
# resource paths
folder_name = 'linear'
current_path = str(pathlib.Path().resolve())
hand_path = current_path+"/resources/2v2_nosensors/2v2_nosensors_limited.urdf"
cube_path = current_path + \
    "/resources/object_models/2v2_mod/2v2_mod_cuboid_small.urdf"
cylinder_path = current_path + \
    "/resources/object_models/2v2_mod/2v2_mod_cylinder_small_alt.urdf"
data_path = current_path+"/data/" + folder_name +'/'
points_path = current_path+"/resources/points.csv"
expert_data_path = current_path+'/resources/episode_all.pkl'


x = [0,0.055, 0.055, 0.055, 0, -0.055, -0.055, -0.055]
y = [0.055, 0.055, 0, -0.055, -0.055, -0.055, 0, 0.055]
pose_list = [[i,j] for i,j in zip(x,y)]
# start pybullet
# physics_client = p.connect(p.GUI)
physics_client = p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)
p.resetDebugVisualizerCamera(cameraDistance=.02, cameraYaw=0, cameraPitch=-89.9999,
                             cameraTargetPosition=[0, 0.1, 0.5])

# load objects into pybullet
plane_id = p.loadURDF("plane.urdf")
hand_id = p.loadURDF(hand_path, useFixedBase=True,
                     basePosition=[0.0, 0.0, 0.05])
cube_id = p.loadURDF(cube_path, basePosition=[0.0, 0.16, .05])
# cylinder_id = p.loadURDF(cylinder_path, basePosition=[0.0, 0.16, .05])
# Create TwoFingerGripper Object and set the initial joint positions
hand = TwoFingerGripper(hand_id, path=hand_path)

p.resetJointState(hand_id, 0, .75)
p.resetJointState(hand_id, 1, -1.4)
p.resetJointState(hand_id, 2, -.75)
p.resetJointState(hand_id, 3, 1.4)

# Create ObjectBase for the cube object
cube = ObjectWithVelocity(cube_id, path=cube_path)
# cylinder = ObjectWithVelocity(cylinder_id, path=cylinder_path)
# cube = ObjectVelocityDF(cube_id, path=cube_path)


# change visual of gripper
p.changeVisualShape(hand_id, -1, rgbaColor=[0.3, 0.3, 0.3, 1])
p.changeVisualShape(hand_id, 0, rgbaColor=[1, 0.5, 0, 1])
p.changeVisualShape(hand_id, 1, rgbaColor=[0.3, 0.3, 0.3, 1])
p.changeVisualShape(hand_id, 2, rgbaColor=[1, 0.5, 0, 1])
p.changeVisualShape(hand_id, 3, rgbaColor=[0.3, 0.3, 0.3, 1])
# p.setTimeStep(1/2400)


goal_poses = GoalHolder(pose_list)
# state and reward
# state = StateDefault(objects=[hand, cube])
state = StateRL(objects=[hand, cube, goal_poses])
# state = StateRL(objects=[hand, cylinder, goal_poses])
action = rl_action.ExpertAction()
reward = rl_reward.ExpertReward()
arg_dict = {'state_dim': 8, 'action_dim': 4, 'max_action': 1.57, 'n': 5, 'discount': 0.995, 'tau': 0.0005,
            'batch_size': 100, 'expert_sampling_proportion': 0.7}


# replay buffer
replay_buffer = ReplayBufferPriority(buffer_size=1000000)
replay_buffer.preload_buffer_PKL(expert_data_path)
# replay_buffer = ReplayBufferDF(state=state, action=action, reward=reward)


# environment and recording
env = rl_env.ExpertEnv(hand=hand, obj=cube)
# env = rl_env.ExpertEnv(hand=hand, obj=cylinder)

# Create phase
manipulation = manipulation_phase_rl.ManipulationRL(
    hand, cube, x, y, state, action, reward, replay_buffer=replay_buffer, args=arg_dict, tbname=folder_name)
# manipulation = manipulation_phase_rl.ManipulationRL(
#     hand, cylinder, x, y, state, action, reward, replay_buffer=replay_buffer, args=arg_dict)
# data recording
record_data = RecordDataRLPKL(
    data_path=data_path, state=state, action=action, reward=reward, save_all=True, controller=manipulation.controller)

# sim manager
manager = SimManagerRL(num_episodes=len(pose_list), env=env, episode=EpisodeDefault(), record_data=record_data, replay_buffer=replay_buffer, state=state, action=action, reward=reward, TensorboardName='test1')

# add phase to sim manager
manager.add_phase("manipulation", manipulation, start=True)

# load up replay buffer
# for i in range(4):
#     manager.run()
#     manager.phase_manager.phase_dict['manipulation'].reset()
#print(p.getClosestPoints(cube.id, hand.id, 1, -1, 1, -1))
# Run the sim
done_training = False
training_length = 200
while not done_training:
    for k in range(training_length):
        manager.run()
        # print('TRAINING NOW')
        # manager.phase_manager.phase_dict["manipulation"].controller.train_policy()
        manager.phase_manager.phase_dict['manipulation'].reset()
    flag = True
    while flag: 
        a = 0#input('input epochs to train for (0 for end)')
        try:
            training_length = int(a)
            flag = False
            if training_length == 0:
                done_training = True
        except ValueError:
            print('input a valid number')

# manager.stall()
