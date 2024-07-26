#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pybullet_data
from demos.rl_demo import multiprocess_env
from demos.rl_demo import multiprocess_manipulation_phase
from demos.rl_demo import multiprocess_reward
# import rl_env
from demos.rl_demo.multiprocess_state import MultiprocessState
from mojograsp.simcore.goal_holder import  GoalHolder, RandomGoalHolder, SimpleGoalHolder, HRLGoalHolder
from demos.rl_demo import rl_action
from demos.rl_demo import multiprocess_direction_reward
from demos.rl_demo import multiprocess_hierarchical_wrapper
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import pandas as pd
from demos.rl_demo.multiprocess_record import MultiprocessRecordData
from mojograsp.simobjects.two_finger_gripper import TwoFingerGripper
from mojograsp.simobjects.object_with_velocity import ObjectWithVelocity
from mojograsp.simcore.priority_replay_buffer import ReplayBufferPriority
import pickle as pkl
import json
from stable_baselines3 import TD3, PPO, DDPG, HerReplayBuffer
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import get_device
import wandb
import numpy as np
import time
import os
import multiprocessing
import json
from scipy.spatial.transform import Rotation as R
import torch
from demos.rl_demo import full_hierarchical_wrapper
# from stable_baselines3.DQN import MlpPolicy

def make_env(arg_dict=None,rank=0,hand_info=None):
    def _init():
        import pybullet as p1
        env, _ = make_pybullet(arg_dict, p1, rank, hand_info)
        return env
    return _init


def make_pybullet(arg_dict, pybullet_instance, rank, hand_info, viz=False):
    # resource paths
    this_path = os.path.abspath(__file__)
    overall_path = os.path.dirname(os.path.dirname(os.path.dirname(this_path)))
    args=arg_dict
    theta = np.random.uniform(0, 2*np.pi,1000)
    r = (1-(np.random.uniform(0, 0.95,1000))**2) * 50/1000
    x = r * np.sin(theta)
    y = r * np.cos(theta)
    obj_pos = np.array([[i,j] for i,j in zip(x,y)])
    theta = np.random.uniform(-15/180*np.pi, 15/180*np.pi,1000)
    fingers = np.random.uniform(0.01,0.01,(2,1000))
    print(obj_pos)
    goals = HRLGoalHolder(obj_pos, fingers, theta)

    # setup pybullet client to either run with or without rendering
    if viz:
        physics_client = pybullet_instance.connect(pybullet_instance.GUI)
    else:
        physics_client = pybullet_instance.connect(pybullet_instance.DIRECT)

    # set initial gravity and general features
    pybullet_instance.setAdditionalSearchPath(pybullet_data.getDataPath())
    pybullet_instance.setGravity(0, 0, -10)
    pybullet_instance.setPhysicsEngineParameter(contactBreakingThreshold=.001)
    pybullet_instance.resetDebugVisualizerCamera(cameraDistance=.02, cameraYaw=0, cameraPitch=-89.9999,
                                 cameraTargetPosition=[0, 0.1, 0.5])
    
    # load hand/hands 
    if rank[1] < len(args['hand_file_list']):
        raise IndexError('TOO MANY HANDS FOR NUMBER OF PROVIDED CORES')
    elif rank[1] % len(args['hand_file_list']) != 0:
        print('WARNING: number of hands does not evenly divide into number of pybullet instances. Hands will have uneven number of samples')
    if args['domain_randomization_object_size']:
        if type(args['object_path']) == str:
            object_path = args['object_path']
            object_key = "small"
            print('older version of object loading, no object domain randomization used')
        else:
            object_path = args['object_path'][rank[0]%len(args['object_path'])]
            if 'add10' in object_path:
                object_key = 'add10'
            elif 'sub10' in object_path:
                object_key = 'sub10'
            else:
                object_key = 'small'
    else:
        if type(args['object_path']) == str:
            object_path = args['object_path']
            object_key = "small"
            # print('older version of object loading, no object domain randomization used')
        else:
            # print('normally would get object DR but not this time')
            object_path = args['object_path'][0]
            object_key = 'small'   

    this_hand = args['hand_file_list'][rank[0]%len(args['hand_file_list'])]
    hand_type = this_hand.split('/')[0]
    hand_keys = hand_type.split('_')
    info_1 = hand_info[hand_keys[-1]][hand_keys[1]]
    info_2 = hand_info[hand_keys[-1]][hand_keys[2]]
    if args['contact_start']:
        hand_param_dict = {"link_lengths":[info_1['link_lengths'],info_2['link_lengths']],
                        "starting_angles":[info_1['contact_start_angles'][object_key][0],info_1['contact_start_angles'][object_key][1],-info_2['contact_start_angles'][object_key][0],-info_2['contact_start_angles'][object_key][1]],
                        "palm_width":info_1['palm_width'],
                        "hand_name":hand_type}
    else:
        # print('STARTING AWAY FROM THE OBJECT')
        hand_param_dict = {"link_lengths":[info_1['link_lengths'],info_2['link_lengths']],
                        "starting_angles":[info_1['near_start_angles'][object_key][0],info_1['near_start_angles'][object_key][1],-info_2['near_start_angles'][object_key][0],-info_2['near_start_angles'][object_key][1]],
                        "palm_width":info_1['palm_width'],
                        "hand_name":hand_type}


    # load objects into pybullet
    plane_id = pybullet_instance.loadURDF("plane.urdf", flags=pybullet_instance.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
    hand_id = pybullet_instance.loadURDF(args['hand_path'] + '/' + this_hand, useFixedBase=True,
                         basePosition=[0.0, 0.0, 0.05], flags=pybullet_instance.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
    obj_id = pybullet_instance.loadURDF(object_path, basePosition=[0.0, 0.10, .05], flags=pybullet_instance.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
    # Create TwoFingerGripper Object and set the initial joint positions
    hand = TwoFingerGripper(hand_id, path=args['hand_path'] + '/' + this_hand,hand_params=hand_param_dict)
    
    # change visual of gripper
    pybullet_instance.changeVisualShape(hand_id, -1, rgbaColor=[0.3, 0.3, 0.3, 1])
    pybullet_instance.changeVisualShape(hand_id, 0, rgbaColor=[1, 0.5, 0, 1])
    pybullet_instance.changeVisualShape(hand_id, 1, rgbaColor=[0.3, 0.3, 0.3, 1])
    pybullet_instance.changeVisualShape(hand_id, 3, rgbaColor=[1, 0.5, 0, 1])
    pybullet_instance.changeVisualShape(hand_id, 4, rgbaColor=[0.3, 0.3, 0.3, 1])
    obj = ObjectWithVelocity(obj_id, path=object_path,name='obj_2')

    # state, action and reward
    state = MultiprocessState(pybullet_instance, objects=[hand, obj, goals], prev_len=args['pv'])
    if args['freq'] ==240:
        action = rl_action.ExpertAction()
    else:
        action = rl_action.InterpAction(args['freq'])
    reward = multiprocess_reward.MultiprocessReward(pybullet_instance)

    #change initial physics parameters
    pybullet_instance.changeDynamics(plane_id,-1,lateralFriction=0.05, spinningFriction=0.05, rollingFriction=0.05)
    pybullet_instance.changeDynamics(obj.id, -1, mass=.03, restitution=.95, lateralFriction=1)
    
    # set up dictionary for manipulation phase
    arg_dict = args.copy()
    if args['action'] == 'Joint Velocity':
        arg_dict['ik_flag'] = False
    else:
        arg_dict['ik_flag'] = True
    

    env = multiprocess_env.MultiprocessSingleShapeEnv(pybullet_instance, hand=hand, obj=obj, hand_type=hand_type, args=args)
    
    # Create phase
    manipulation = multiprocess_manipulation_phase.MultiprocessManipulation(
        hand, obj, state, action, reward, env, args=arg_dict, hand_type=hand_type)
    
    # data recording
    record_data = MultiprocessRecordData(rank,
        data_path=args['save_path'], state=state, action=action, reward=reward, save_all=False, controller=manipulation.controller)
    
    # gym wrapper around pybullet environment
    gym_env = full_hierarchical_wrapper.FullTaskWrapper(env, manipulation, record_data, args)
    return gym_env, args

def main(filepath = None, train_type='pre'):
    num_cpu = 2#multiprocessing.cpu_count() # Number of processes to use
    # Create the vectorized environment
    from hbaselines.algorithms import RLAlgorithm
    from hbaselines.goal_conditioned.td3 import GoalConditionedPolicy
    with open(filepath, 'r') as argfile:
        args = json.load(argfile)
    key_file = os.path.abspath(__file__)
    key_file = os.path.dirname(key_file)
    key_file = os.path.join(key_file,'resources','hand_bank','hand_params.json')
    with open(key_file,'r') as hand_file:
        hand_params = json.load(hand_file)
    # vec_env = SubprocVecEnv([make_env(args,[i,num_cpu],hand_info=hand_params) for i in range(num_cpu)]) 
    # vec_env.horizon = 25
    import pybullet as p1
    env,_ = make_pybullet(args, p1, [0,1],hand_info=hand_params)
    env.horizon = 25
    train_timesteps = int(args['evaluate']*(args['tsteps']+1)/num_cpu)
    callback = multiprocess_hierarchical_wrapper.MultiEvaluateCallback(env,n_eval_episodes=int(1200), eval_freq=train_timesteps, best_model_save_path=args['save_path'])
    model = RLAlgorithm(policy=GoalConditionedPolicy, total_steps=10,
                        env=env,policy_kwargs={'env_name':'FullHandTask'})
    try:
        model.learn('testing_thing')#,callback=callback)
        # print(time.time()-a)
        filename = os.path.dirname(filepath)
        model.save(filename+'/last_model_full')

    except KeyboardInterrupt:
        filename = os.path.dirname(filepath)
        model.save(filename+'/canceled_model_full')

    
if __name__ == '__main__':
    main('./data/hrl_test_3/experiment_config.json', 'feudal')
    # evaluate('./data/HRL_test_1/experiment_config.json')