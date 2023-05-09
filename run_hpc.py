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
import time
import sys
import os

def run_pybullet(filepath, window=None, runtype='run'):
    # resource paths
    with open(filepath, 'r') as argfile:
        args = json.load(argfile)
    
    if (args['task'] == 'asterisk'):
        x = [0.03, 0, -0.03, -0.04, -0.03, 0, 0.03, 0.04]
        y = [-0.03, -0.04, -0.03, 0, 0.03, 0.04, 0.03, 0]
    elif args['task'] == 'random' and runtype != 'eval':
        df = pd.read_csv(args['points_path'], index_col=False)
        x = df["x"]
        y = df["y"]
    elif runtype=='eval':
        # df = pd.read_csv('/home/orochi/mojo/mojo-grasp/demos/rl_demo/resources/test_points.csv', index_col=False)
        # print('EVALUATING BOOOIIII')
        # x = df["x"]
        # y = df["y"]
        x = [0.03, 0, -0.03, -0.04, -0.03, 0, 0.03, 0.04]
        y = [-0.03, -0.04, -0.03, 0, 0.03, 0.04, 0.03, 0]
        
    pose_list = [[i,j] for i,j in zip(x,y)]
    
    print(args)
    try:
        if (args['viz']) | (runtype=='eval'):
            physics_client = p.connect(p.GUI)
        else:
            physics_client = p.connect(p.DIRECT)
    except KeyError:
        physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -10)
    p.resetDebugVisualizerCamera(cameraDistance=.02, cameraYaw=0, cameraPitch=-89.9999,
                                 cameraTargetPosition=[0, 0.1, 0.5])
    
    # load objects into pybullet
    plane_id = p.loadURDF("plane.urdf", flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
    hand_id = p.loadURDF(args['hand_path'], useFixedBase=True,
                         basePosition=[0.0, 0.0, 0.05], flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
    obj_id = p.loadURDF(args['object_path'], basePosition=[0.0, 0.10, .05], flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
    
    # Create TwoFingerGripper Object and set the initial joint positions
    hand = TwoFingerGripper(hand_id, path=args['hand_path'])
    
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
    obj = ObjectWithVelocity(obj_id, path=args['object_path'])
    # p.addUserDebugPoints([[0.2,0.1,0.0],[1,0,0]],[[1,0.0,0],[0.5,0.5,0.5]], 1)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    goal_poses = GoalHolder(pose_list)
    # time.sleep(10)
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
    
    # add phase to sim manager
    manager.add_phase("manipulation", manipulation, start=True)
    
    # Run the sim
    if runtype == 'run':
        for k in range(int(args['epochs']/len(x))):
            manager.run()
            manager.phase_manager.phase_dict['manipulation'].reset()
            '''
            if k % args['evaluate'] == 0:
                manager.evaluate()
                manager.phase_manager.phase_dict['manipulation'].reset()
                # manager.train()
            '''
                
        manager.save_network(args['save_path']+'policy')
        # replay_buffer.save_sampling('/home/orochi/mojo/mojo-grasp/demos/rl_demo/data/hard_random_sampling/sampling')
        print('training done, creating episode_all')
        d = data_processor(args['save_path'])
        d.load_data()
        d.save_all()
        manager.phase_manager.phase_dict['manipulation'].controller.policy.save_sampling()
    elif runtype == 'eval':
        manipulation.load_policy(args['save_path']+'policy')
        manager.evaluate()
        manager.phase_manager.phase_dict['manipulation'].reset()
    elif runtype == 'cont':
        manipulation.load_policy(args['save_path']+'policy')
        for k in range(int(args['epochs']/len(x))):
            manager.run()
            manager.phase_manager.phase_dict['manipulation'].reset()
        manager.save_network(args['save_path']+'policy_2')
        print('training done, creating episode_all')
        d = data_processor(args['save_path'])
        d.load_data()
        d.save_all()
    elif runtype == 'transfer':
        manipulation.load_policy(args['load_path']+'policy')
        manager.evaluate()
        manager.phase_manager.phase_dict['manipulation'].reset()
        for k in range(int(args['epochs']/len(x))):
            manager.run()
            manager.phase_manager.phase_dict['manipulation'].reset()
            if k % 10 == 9:
                manager.evaluate()
                manager.phase_manager.phase_dict['manipulation'].reset()
                
def main(run_id):
    print(run_id)
    folder_names = ['0_ftp_control','1_ftp_no_contact','2_ftp_no_contact_velocity','3_ftp_velocity','4_ftp_velocity_x1',
                    '5_ftp_velocity_x2','6_ftp_velocity_x3','7_ftp_velocity_x4','8_ftp_rollout_10','9_ftp_rollout_1']
    
    overall_path = pathlib.Path(__file__).parent.resolve()
    run_path = overall_path.joinpath('demos/rl_demo/data/HPC_runs')
    final_path = run_path.joinpath(folder_names[run_id-1])
    print(str(final_path))
    run_pybullet(str(final_path) + '/experiment_config.json',runtype='run')
    # run_pybullet('/home/orochi/mojo/mojo-grasp/demos/rl_demo/data/6_ik_kegan_point_split/experiment_config.json',runtype='eval')
    # run_pybullet('/home/orochi/mojo/mojo-grasp/demos/rl_demo/data/a_throwaway/experiment_config.json',runtype='run')
    print('buttz')
if __name__ == '__main__':
    # print(sys.argv)
    main(int(sys.argv[1]))
