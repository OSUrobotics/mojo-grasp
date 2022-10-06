#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 14:44:46 2022

@author: orochi
"""
import pybullet as p
import time
import pybullet_data
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
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-10)
planeId = p.loadURDF("plane.urdf")
startPos = [0,0,1]
startOrientation = p.getQuaternionFromEuler([0,0,0])
# cheetahId = p.loadURDF("/minic_cheetah/mini_cheetah.urdf", startPos, startOrientation)
# cheetahId = p.loadURDF("./mini_cheetah/mini_cheetah.urdf", startPos, startOrientation)
cheetahId = p.loadMJCF('./mjcf/half_cheetah.xml')

state = StateRL(objects=[cheetahId])
action = rl_action.ExpertAction()
reward = rl_reward.ExpertReward()
arg_dict = {'state_dim': 14, 'action_dim': 4, 'max_action': 1.57, 'n': 5, 'discount': 0.995, 'tau': 0.0005,
            'batch_size': 100, 'expert_sampling_proportion': 0.7}


# replay buffer
replay_buffer = ReplayBufferDefault(buffer_size=400000, state=state, action=action, reward=reward)
# replay_buffer = ReplayBufferDF(state=state, action=action, reward=reward)

# Run the sim
done_training = False
training_length = 100
while not done_training:
    for k in range(training_length):
        
p.disconnect()