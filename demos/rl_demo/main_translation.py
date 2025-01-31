#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 14:27:04 2022

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
import numpy as np
# resource paths
current_path = str(pathlib.Path().resolve())
hand_path = current_path+"/resources/2v2_nosensors/2v2_nosensors_limited.urdf"
cube_path = current_path + \
    "/resources/object_models/2v2_mod/2v2_mod_cuboid_small.urdf"
data_path = current_path+"/data/coords/one-dir"
points_path = current_path+"/resources/points.csv"

# Load in the cube goal positions
#df = pd.read_csv(points_path, index_col=False)
#x = df["x"]
#y = df["y"]
#
#length = np.sqrt(np.random.uniform(0, 0.0036, 100))
#angle = np.pi * np.random.uniform(0, 2, 100)
#
#x = length * np.cos(angle)
#y = length * np.sin(angle)
# coords = [[-0.055, 0.0],[-0.055, 0.055],[0.0, 0.055],[0.055, 0.0555],[0.055, 0.0],[0.055,-0.055],[0.0, -0.055],[-0.055, -0.055]]
coords = [[-0.055, -0.055]]
x = [i[0] for i in coords]
y = [i[1] for i in coords]
# start pybullet
physics_client = p.connect(p.GUI)
# physics_client = p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)
p.resetDebugVisualizerCamera(cameraDistance=.02, cameraYaw=0, cameraPitch=-89.9999,
                             cameraTargetPosition=[0, 0.1, 0.5])

# load objects into pybullet
plane_id = p.loadURDF("plane.urdf")
hand_id = p.loadURDF(hand_path, useFixedBase=True,
                     basePosition=[0.0, 0.0, 0.05])
cube_id = p.loadURDF(cube_path, basePosition=[0.0, 0.16, .05])
# Create TwoFingerGripper Object and set the initial joint positions
hand = TwoFingerGripper(hand_id, path=hand_path)

p.resetJointState(hand_id, 0, .75)
p.resetJointState(hand_id, 1, -1.4)
p.resetJointState(hand_id, 2, -.75)
p.resetJointState(hand_id, 3, 1.4)

# Create ObjectBase for the cube object
cube = ObjectWithVelocity(cube_id, path=cube_path)
# cube = ObjectVelocityDF(cube_id, path=cube_path)


# change visual of gripper
p.changeVisualShape(hand_id, -1, rgbaColor=[0.3, 0.3, 0.3, 1])
p.changeVisualShape(hand_id, 0, rgbaColor=[1, 0.5, 0, 1])
p.changeVisualShape(hand_id, 1, rgbaColor=[0.3, 0.3, 0.3, 1])
p.changeVisualShape(hand_id, 2, rgbaColor=[1, 0.5, 0, 1])
p.changeVisualShape(hand_id, 3, rgbaColor=[0.3, 0.3, 0.3, 1])
# p.setTimeStep(1/2400)


goal_poses = GoalHolder(coords)
# state and reward
# state = StateDefault(objects=[hand, cube])
state = StateRL(objects=[hand, cube, goal_poses])
action = rl_action.ExpertAction()
reward = rl_reward.TranslateReward()
arg_dict = {'state_dim': 16, 'action_dim': 4, 'max_action': 1.57, 'n': 5, 'discount': 0.995, 'tau': 0.0005,
            'batch_size': 100, 'expert_sampling_proportion': 0.7, "pred_dim": 2}


# replay buffer
replay_buffer = ReplayBufferDefault(buffer_size=400000, state=state, action=action, reward=reward)
# replay_buffer = ReplayBufferDF(state=state, action=action, reward=reward)


# environment and recording
env = rl_env.ExpertEnv(hand=hand, obj=cube)

# Create phase
manipulation = manipulation_phase_rl.ManipulationRLTranslate(
    hand, cube, x, y, state, action, reward, replay_buffer=replay_buffer, args=arg_dict)

# data recording
record_data = RecordDataRLPKL(
    data_path=data_path, state=state, action=action, reward=reward, save_all=True, controller=manipulation.controller)

# sim manager
manager = SimManagerRL(num_episodes=len(coords), env=env, record_data=record_data, replay_buffer=replay_buffer)

# add phase to sim manager
manager.add_phase("manipulation", manipulation, start=True)

# load up replay buffer
# for i in range(4):
#     manager.run()
#     manager.phase_manager.phase_dict['manipulation'].reset()
#print(p.getClosestPoints(cube.id, hand.id, 1, -1, 1, -1))
# Run the sim
done_training = False
training_length = 1000
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
