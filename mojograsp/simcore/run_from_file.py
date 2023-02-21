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


def run_pybullet(filepath, window=None):
    # resource paths
    with open(filepath, 'rb') as argfile:
        args = pkl.load(argfile)
    
    # expert_angles
    # [-.695, 1.487, 0.695, -1.487]
    
    # expert object pose
    # [0.0, .1067, .05]
    if args['task'] == 'asterisk':
        x = [0.03, 0, -0.03, -0.04, -0.03, 0, 0.03, 0.04]
        y = [-0.03, -0.04, -0.03, 0, 0.03, 0.04, 0.03, 0]
    elif args['task'] == 'random':
        df = pd.read_csv(args['points_path'], index_col=False)
        
        x = df["x"]
        y = df["y"]
    # x = [0,0.055, 0.055, 0.055, 0, -0.055, -0.055, -0.055]
    # y = [0.055, 0.055, 0, -0.055, -0.055, -0.055, 0, 0.055]
    # print(len(x))
    pose_list = [[i,j] for i,j in zip(x,y)]
    # start pybullet
    print(args)
    
    # physics_client = p.connect(p.GUI)
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
    
    # p.resetJointState(hand_id, 0, .695)
    # p.resetJointState(hand_id, 1, -1.487)
    # p.resetJointState(hand_id, 3, -.695)
    # p.resetJointState(hand_id, 4, 1.487)
    
    # cylinder = ObjectWithVelocity(cylinder_id, path=cylinder_path)
    # obj = ObjectVelocityDF(obj_id, path=args['object_path'])
    
    
    # change visual of gripper
    p.changeVisualShape(hand_id, 0, rgbaColor=[0.3, 0.3, 0.3, 1])
    p.changeVisualShape(hand_id, 1, rgbaColor=[1, 0.5, 0, 1])
    p.changeVisualShape(hand_id, 3, rgbaColor=[0.3, 0.3, 0.3, 1])
    p.changeVisualShape(hand_id, 4, rgbaColor=[1, 0.5, 0, 1])
    p.changeVisualShape(hand_id, -1, rgbaColor=[0.3, 0.3, 0.3, 1])
    # p.setTimeStep(1/2400)
    
    obj = ObjectWithVelocity(obj_id, path=args['object_path'])
    
    goal_poses = GoalHolder(pose_list)
    # state and reward
    # state = StateDefault(objects=[hand, obj])
    state = StateRL(objects=[hand, obj, goal_poses])
    # state = StateRL(objects=[hand, cylinder, goal_poses])
    action = rl_action.ExpertAction()
    reward = rl_reward.ExpertReward()
    if args['action'] == 'Joint Velocity':
        ik_flag = False
    else:
        ik_flag = True
    # print(args['state_dim'])
    arg_dict = {'state_dim': args['state_dim'], 'action_dim': 4, 'max_action': args['max_action'],
                'n': 5, 'discount': args['discount'], 'tau': 0.0005,'batch_size': args['batch_size'], 
                'epsilon':args['epsilon'], 'edecay': args['edecay'], 'ik_flag': ik_flag,
                'reward':args['reward'], 'model':args['model'], 'tname':args['tname']}
    # arg_dict = {'state_dim': 8, 'action_dim': 4, 'max_action': 0.005, 'n': 5, 'discount': 0.995, 'tau': 0.0005,
    #             'batch_size': 100, 'expert_sampling_proportion': 0.7}
    
    # replay buffer
    replay_buffer = ReplayBufferPriority(buffer_size=4080000)
    # replay_buffer.preload_buffer_PKL(args['edata'])
    # replay_buffer = ReplayBufferDF(state=state, action=action, reward=reward)
    
    
    # environment and recording
    env = rl_env.ExpertEnv(hand=hand, obj=obj)
    # env = rl_env.ExpertEnv(hand=hand, obj=cylinder)
    
    # Create phase
    manipulation = manipulation_phase_rl.ManipulationRL(
        hand, obj, x, y, state, action, reward, replay_buffer=replay_buffer, args=arg_dict)
    # manipulation = manipulation_phase_rl.ManipulationRL(
    #     hand, cylinder, x, y, state, action, reward, replay_buffer=replay_buffer, args=arg_dict)
    # data recording
    record_data = RecordDataRLPKL(
        data_path=args['save_path'], state=state, action=action, reward=reward, save_all=False, controller=manipulation.controller)
    
    # sim manager
    manager = SimManagerRLHER(num_episodes=len(pose_list), env=env, episode=EpisodeDefault(), record_data=record_data, replay_buffer=replay_buffer, state=state, action=action, reward=reward, args=arg_dict)
    
    # add phase to sim manager
    manager.add_phase("manipulation", manipulation, start=True)
    
    # load up replay buffer
    # for i in range(4):
    #     manager.run()
    #     manager.phase_manager.phase_dict['manipulation'].reset()
    #print(p.getClosestPoints(obj.id, hand.id, 1, -1, 1, -1))
    # Run the sim
    for k in range(int(args['epochs']/len(x))):
        # window.write_event_value('update_progress_1', k/int(args['epochs']/len(x)))
        manager.run()
        # print('TRAINING NOW')
        # manager.phase_manager.phase_dict["manipulation"].controller.train_policy()
        manager.phase_manager.phase_dict['manipulation'].reset()
            
        
    print('training done, creating episode_all')
    d = data_processor(args['save_path'])
    d.load_data()
    d.save_all()
    manager.save_network(args['save_path']+'policy')
    # manager.stall()


def evaluate_pybullet(filepath, window=None):
    # resource paths
    with open(filepath, 'rb') as argfile:
        args = pkl.load(argfile)
    
    # expert_angles
    # [-.695, 1.487, 0.695, -1.487]
    
    # expert object pose
    # [0.0, .1067, .05]
    if args['task'] == 'asterisk':
        x = [0.03, 0, -0.03, -0.04, -0.03, 0, 0.03, 0.04]
        y = [-0.03, -0.04, -0.03, 0, 0.03, 0.04, 0.03, 0]
    elif args['task'] == 'random':
        df = pd.read_csv(args['points_path'], index_col=False)
        
        x = df["x"]
        y = df["y"]
    # x = [0,0.055, 0.055, 0.055, 0, -0.055, -0.055, -0.055]
    # y = [0.055, 0.055, 0, -0.055, -0.055, -0.055, 0, 0.055]
    # print(len(x))
    pose_list = [[i,j] for i,j in zip(x,y)]
    # start pybullet
    print(args)
    
    # physics_client = p.connect(p.GUI)
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
    
    # p.resetJointState(hand_id, 0, .695)
    # p.resetJointState(hand_id, 1, -1.487)
    # p.resetJointState(hand_id, 3, -.695)
    # p.resetJointState(hand_id, 4, 1.487)
    
    # cylinder = ObjectWithVelocity(cylinder_id, path=cylinder_path)
    # obj = ObjectVelocityDF(obj_id, path=args['object_path'])
    
    
    # change visual of gripper
    p.changeVisualShape(hand_id, 0, rgbaColor=[0.3, 0.3, 0.3, 1])
    p.changeVisualShape(hand_id, 1, rgbaColor=[1, 0.5, 0, 1])
    p.changeVisualShape(hand_id, 3, rgbaColor=[0.3, 0.3, 0.3, 1])
    p.changeVisualShape(hand_id, 4, rgbaColor=[1, 0.5, 0, 1])
    p.changeVisualShape(hand_id, -1, rgbaColor=[0.3, 0.3, 0.3, 1])
    # p.setTimeStep(1/2400)
    
    obj = ObjectWithVelocity(obj_id, path=args['object_path'])
    
    goal_poses = GoalHolder(pose_list)
    # state and reward
    # state = StateDefault(objects=[hand, obj])
    state = StateRL(objects=[hand, obj, goal_poses])
    # state = StateRL(objects=[hand, cylinder, goal_poses])
    action = rl_action.ExpertAction()
    reward = rl_reward.ExpertReward()
    if args['action'] == 'Joint Velocity':
        ik_flag = False
    else:
        ik_flag = True
    arg_dict = {'state_dim': args['state_dim'], 'action_dim': 4, 'max_action': args['max_action'],
                'n': 5, 'discount': args['discount'], 'tau': 0.0005,'batch_size': args['batch_size'], 
                'epsilon':args['epsilon'], 'edecay': args['edecay'], 'ik_flag': ik_flag,
                'reward':args['reward'], 'model':args['model'], 'tname':args['tname']}
    # arg_dict = {'state_dim': 8, 'action_dim': 4, 'max_action': 0.005, 'n': 5, 'discount': 0.995, 'tau': 0.0005,
    #             'batch_size': 100, 'expert_sampling_proportion': 0.7}
    
    # replay buffer
    replay_buffer = ReplayBufferPriority(buffer_size=4080000)
    # replay_buffer.preload_buffer_PKL(args['edata'])
    # replay_buffer = ReplayBufferDF(state=state, action=action, reward=reward)
    
    
    # environment and recording
    env = rl_env.ExpertEnv(hand=hand, obj=obj)
    # env = rl_env.ExpertEnv(hand=hand, obj=cylinder)
    
    # Create phase
    manipulation = manipulation_phase_rl.ManipulationRL(
        hand, obj, x, y, state, action, reward, replay_buffer=replay_buffer, args=arg_dict)
    
    manipulation.load_policy(args['save_path']+'policy_2')
    # manipulation = manipulation_phase_rl.ManipulationRL(
    #     hand, cylinder, x, y, state, action, reward, replay_buffer=replay_buffer, args=arg_dict)
    # data recording
    record_data = RecordDataRLPKL(
        data_path=args['save_path'], state=state, action=action, reward=reward, save_all=False, controller=manipulation.controller)
    
    # sim manager
    manager = SimManagerRLHER(num_episodes=len(pose_list), env=env, episode=EpisodeDefault(), record_data=record_data, replay_buffer=replay_buffer, state=state, action=action, reward=reward, args=arg_dict)
    
    # add phase to sim manager
    manager.add_phase("manipulation", manipulation, start=True)
    # manager.record_video = True
    manager.evaluate()
    manager.phase_manager.phase_dict['manipulation'].reset()
            
        
    # print('training done, creating episode_all')
    # d = data_processor(args['save_path'])
    # d.load_data()
    # d.save_all()
    # manager.save_network(args['save_path']+'policy')
    # manager.stall()
    
def continue_pybullet(filepath, window=None):
    # resource paths
    with open(filepath, 'rb') as argfile:
        args = pkl.load(argfile)
    
    # expert_angles
    # [-.695, 1.487, 0.695, -1.487]
    
    # expert object pose
    # [0.0, .1067, .05]
    if args['task'] == 'asterisk':
        x = [0.03, 0, -0.03, -0.04, -0.03, 0, 0.03, 0.04]
        y = [-0.03, -0.04, -0.03, 0, 0.03, 0.04, 0.03, 0]
    elif args['task'] == 'random':
        df = pd.read_csv(args['points_path'], index_col=False)
        
        x = df["x"]
        y = df["y"]
    # x = [0,0.055, 0.055, 0.055, 0, -0.055, -0.055, -0.055]
    # y = [0.055, 0.055, 0, -0.055, -0.055, -0.055, 0, 0.055]
    # print(len(x))
    pose_list = [[i,j] for i,j in zip(x,y)]
    # start pybullet
    print(args)
    
    # physics_client = p.connect(p.GUI)
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
    
    # p.resetJointState(hand_id, 0, .695)
    # p.resetJointState(hand_id, 1, -1.487)
    # p.resetJointState(hand_id, 3, -.695)
    # p.resetJointState(hand_id, 4, 1.487)
    
    # cylinder = ObjectWithVelocity(cylinder_id, path=cylinder_path)
    # obj = ObjectVelocityDF(obj_id, path=args['object_path'])
    
    
    # change visual of gripper
    p.changeVisualShape(hand_id, 0, rgbaColor=[0.3, 0.3, 0.3, 1])
    p.changeVisualShape(hand_id, 1, rgbaColor=[1, 0.5, 0, 1])
    p.changeVisualShape(hand_id, 3, rgbaColor=[0.3, 0.3, 0.3, 1])
    p.changeVisualShape(hand_id, 4, rgbaColor=[1, 0.5, 0, 1])
    p.changeVisualShape(hand_id, -1, rgbaColor=[0.3, 0.3, 0.3, 1])
    # p.setTimeStep(1/2400)
    
    obj = ObjectWithVelocity(obj_id, path=args['object_path'])
    
    goal_poses = GoalHolder(pose_list)
    # state and reward
    # state = StateDefault(objects=[hand, obj])
    state = StateRL(objects=[hand, obj, goal_poses])
    # state = StateRL(objects=[hand, cylinder, goal_poses])
    action = rl_action.ExpertAction()
    reward = rl_reward.ExpertReward()
    if args['action'] == 'Joint Velocity':
        ik_flag = False
    else:
        ik_flag = True
    arg_dict = {'state_dim': args['state_dim'], 'action_dim': 4, 'max_action': args['max_action'],
                'n': 5, 'discount': args['discount'], 'tau': 0.0005,'batch_size': args['batch_size'], 
                'epsilon':args['epsilon']*args['edecay']**args['epochs'], 'edecay': args['edecay'], 'ik_flag': ik_flag,
                'reward':args['reward'], 'model':args['model'], 'tname':args['tname']}
    # arg_dict = {'state_dim': 8, 'action_dim': 4, 'max_action': 0.005, 'n': 5, 'discount': 0.995, 'tau': 0.0005,
    #             'batch_size': 100, 'expert_sampling_proportion': 0.7}
    
    # replay buffer
    replay_buffer = ReplayBufferPriority(buffer_size=4080000)
    # replay_buffer.preload_buffer_PKL(args['edata'])
    # replay_buffer = ReplayBufferDF(state=state, action=action, reward=reward)
    
    
    # environment and recording
    env = rl_env.ExpertEnv(hand=hand, obj=obj)
    # env = rl_env.ExpertEnv(hand=hand, obj=cylinder)
    
    # Create phase
    manipulation = manipulation_phase_rl.ManipulationRL(
        hand, obj, x, y, state, action, reward, replay_buffer=replay_buffer, args=arg_dict)
    # manipulation = manipulation_phase_rl.ManipulationRL(
    #     hand, cylinder, x, y, state, action, reward, replay_buffer=replay_buffer, args=arg_dict)
    # data recording
    record_data = RecordDataRLPKL(
        data_path=args['save_path'], state=state, action=action, reward=reward, save_all=False, controller=manipulation.controller)
    
    # sim manager
    manager = SimManagerRLHER(num_episodes=len(pose_list), env=env, episode=EpisodeDefault(), record_data=record_data, replay_buffer=replay_buffer, state=state, action=action, reward=reward, args=arg_dict)
    
    # add phase to sim manager
    manager.add_phase("manipulation", manipulation, start=True)
    
    manipulation.load_policy(args['save_path']+'policy')
    # load up replay buffer
    # for i in range(4):
    #     manager.run()
    #     manager.phase_manager.phase_dict['manipulation'].reset()
    #print(p.getClosestPoints(obj.id, hand.id, 1, -1, 1, -1))
    # Run the sim
    for k in range(int(args['epochs']/len(x))):
        # window.write_event_value('update_progress_1', k/int(args['epochs']/len(x)))
        manager.run()
        # print('TRAINING NOW')
        # manager.phase_manager.phase_dict["manipulation"].controller.train_policy()
        manager.phase_manager.phase_dict['manipulation'].reset()
            
        
    print('training done, creating episode_all')
    d = data_processor(args['save_path'])
    d.load_data()
    d.save_all()
    manager.save_network(args['save_path']+'policy_2')
    # manager.stall()

def main():
    # evaluate_pybullet('/home/orochi/mojo/mojo-grasp/demos/rl_demo/data/reward_based_epsilon/experiment_config.pkl')
    # continue_pybullet('/home/orochi/mojo/mojo-grasp/demos/rl_demo/data/reward_based_epsilon/experiment_config.pkl')
    run_pybullet('/home/orochi/mojo/mojo-grasp/demos/rl_demo/data/base_location_too/experiment_config.pkl')
if __name__ == '__main__':
    main()