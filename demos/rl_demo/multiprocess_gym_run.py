#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 10:53:58 2023

@author: orochi
"""

from pybullet_utils import bullet_client as bc
import pybullet_data
from demos.rl_demo import multiprocess_env
from demos.rl_demo import multiprocess_manipulation_phase
# import rl_env
from demos.rl_demo.multiprocess_state import MultiprocessState, GoalHolder, RandomGoalHolder
from demos.rl_demo import rl_action
from demos.rl_demo import multiprocess_reward
from demos.rl_demo import multiprocess_gym_wrapper
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
import wandb
import numpy as np
import time
import os
import multiprocessing


# from stable_baselines3.DQN import MlpPolicy

def make_env(filepath=None,rank=0):
    def _init():
        import pybullet as p1
        env, _, _ = make_pybullet(filepath, p1, rank)
        return env
    return _init

def make_pybullet(filepath, pybullet_instance, rank):
    # resource paths
    this_path = os.path.abspath(__file__)
    overall_path = os.path.dirname(os.path.dirname(os.path.dirname(this_path)))
    with open(filepath, 'r') as argfile:
        args = json.load(argfile)
    # print(args['task'])
    if args['task'] == 'asterisk':
        x = [0.03, 0, -0.03, -0.04, -0.03, 0, 0.03, 0.04]
        y = [-0.03, -0.04, -0.03, 0, 0.03, 0.04, 0.03, 0]
        xeval = x
        yeval = y
        eval_names = ['SE','S','SW','W','NW','N','NE','E'] 
    elif 'random' == args['task']:
        df = pd.read_csv(args['points_path'], index_col=False)
        x = df["x"]
        y = df["y"]
        df2 = pd.read_csv(overall_path + '/demos/rl_demo/resources/test_points.csv', index_col=False)
        xeval = df2['x']
        yeval = df2['y']
    elif ('big_random' == args['task']) | ('multi' == args['task']):
        df = pd.read_csv(args['points_path'], index_col=False)
        x = df["x"]
        y = df["y"]
        df2 = pd.read_csv(overall_path + '/demos/rl_demo/resources/test_points_big.csv', index_col=False)
        xeval = df2['x']
        yeval = df2['y']
    elif 'full_random' == args['task']:
        df = pd.read_csv(args['points_path'], index_col=False)
        x = df["x"]
        y = df["y"]
        xeval = [0.045, 0, -0.045, -0.06, -0.045, 0, 0.045, 0.06]
        yeval = [-0.045, -0.06, -0.045, 0, 0.045, 0.06, 0.045, 0]
        eval_names = ['SE','S','SW','W','NW','N','NE','E'] 
    elif args['task'] == 'unplanned_random':
        x = [0.02]
        y = [0.065]
        df2 = pd.read_csv(overall_path + '/demos/rl_demo/resources/points.csv', index_col=False)
        xeval = df2["x"]
        yeval = df2["x"]
        eval_names = 500 * ['Eval']
    elif 'wedge' in args['task']:
        df = pd.read_csv(args['points_path'], index_col=False)
        x = df["x"]
        y = df["y"]
        xeval = x
        yeval = y
    elif args['task'] == 'forward':
        x= [0.0]
        y = [0.04]
        xeval = x
        yeval = y
        eval_names = ['N'] 
    elif args['task'] == 'backward':
        x= [0.0]
        y = [-0.04]
        xeval = x
        yeval = y
        eval_names = ['S'] 
    elif args['task'] == 'left':
        x= [-0.04]
        y = [0.0]
        xeval = x
        yeval = y
        eval_names = ['W'] 
    elif args['task'] == 'right':
        x= [0.04]
        y = [0.0]
        xeval = x
        yeval = y
        eval_names = ['E'] 
    elif args['task'] == 'forward_left':
        x= [-0.03]
        y = [0.03]
        xeval = x
        yeval = y
        eval_names = ['NW'] 
    elif args['task'] == 'forward_right':
        x= [0.03]
        y = [0.03]
        xeval = x
        yeval = y
        eval_names = ['NE'] 
    elif args['task'] == 'backward_left':
        x= [-0.03]
        y = [-0.03]
        xeval = x
        yeval = y
        eval_names = ['SW'] 
    elif args['task'] == 'backward_right':
        x= [0.03]
        y = [-0.03]
        xeval = x
        yeval = y
        eval_names = ['SE'] 
    else:
        df = pd.read_csv(args['points_path'], index_col=False)
        print('EVALUATING BOOOIIII')
        x = df["x"]
        y = df["y"]
        xeval = x
        yeval = y
        # xeval = [0.045, 0, -0.045, -0.06, -0.045, 0, 0.045, 0.06]
        # yeval = [-0.045, -0.06, -0.045, 0, 0.045, 0.06, 0.045, 0]
        eval_names = ['eval']*500 
    

    names = ['AsteriskSE.pkl','AsteriskS.pkl','AsteriskSW.pkl','AsteriskW.pkl','AsteriskNW.pkl','AsteriskN.pkl','AsteriskNE.pkl','AsteriskE.pkl']
    pose_list = [[i,j] for i,j in zip(x,y)]
    eval_pose_list = [[i,j] for i,j in zip(xeval,yeval)]
    num_eval = len(eval_pose_list)
    eval_pose_list = eval_pose_list[int(num_eval*rank[0]/rank[1]):int(num_eval*(rank[0]+1)/rank[1])]
    # print(args)
    physics_client = pybullet_instance.connect(pybullet_instance.DIRECT)
    pybullet_instance.setAdditionalSearchPath(pybullet_data.getDataPath())
    pybullet_instance.setGravity(0, 0, -10)
    pybullet_instance.setPhysicsEngineParameter(contactBreakingThreshold=.001)
    pybullet_instance.resetDebugVisualizerCamera(cameraDistance=.02, cameraYaw=0, cameraPitch=-89.9999,
                                 cameraTargetPosition=[0, 0.1, 0.5])
    
    if rank[1] < len(args['hand_file_list']):
        raise IndexError('TOO MANY HANDS FOR NUMBER OF PROVIDED CORES')
    elif rank[1] % len(args['hand_file_list']) != 0:
        print('WARNING: number of hands does not evenly divide into number of pybullet instances. Hands will have uneven number of samples')
    
    this_hand = args['hand_file_list'][rank[1]%len(args['hand_file_list'])]
    # load objects into pybullet
    plane_id = pybullet_instance.loadURDF("plane.urdf", flags=pybullet_instance.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
    hand_id = pybullet_instance.loadURDF(args['hand_path'] + '/' + this_hand, useFixedBase=True,
                         basePosition=[0.0, 0.0, 0.05], flags=pybullet_instance.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
    obj_id = pybullet_instance.loadURDF(args['object_path'], basePosition=[0.0, 0.10, .05], flags=pybullet_instance.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
    
    # Create TwoFingerGripper Object and set the initial joint positions
    hand = TwoFingerGripper(hand_id, path=args['hand_path'] + '/' + this_hand)
    
    # change visual of gripper
    pybullet_instance.changeVisualShape(hand_id, -1, rgbaColor=[0.3, 0.3, 0.3, 1])
    pybullet_instance.changeVisualShape(hand_id, 0, rgbaColor=[1, 0.5, 0, 1])
    pybullet_instance.changeVisualShape(hand_id, 1, rgbaColor=[0.3, 0.3, 0.3, 1])
    pybullet_instance.changeVisualShape(hand_id, 3, rgbaColor=[1, 0.5, 0, 1])
    pybullet_instance.changeVisualShape(hand_id, 4, rgbaColor=[0.3, 0.3, 0.3, 1])
    obj = ObjectWithVelocity(obj_id, path=args['object_path'],name='obj_2')
    
    # For standard loaded goal poses
    if args['task'] == 'unplanned_random':
        goal_poses = RandomGoalHolder([0.02,0.065])
    else:    
        goal_poses = GoalHolder(pose_list)
    try:
        eval_goal_poses = GoalHolder(eval_pose_list,eval_names)
    except NameError:
        # print('No names')
        eval_goal_poses = GoalHolder(eval_pose_list)
    
    # time.sleep(10)
    # state, action and reward
    state = MultiprocessState(pybullet_instance, objects=[hand, obj, goal_poses], prev_len=args['pv'],eval_goals = eval_goal_poses)
    
    if args['freq'] ==240:
        action = rl_action.ExpertAction()
    else:
        action = rl_action.InterpAction(args['freq'])
    
    reward = multiprocess_reward.MultiprocessReward(pybullet_instance)
    pybullet_instance.changeDynamics(plane_id,-1,lateralFriction=0.05, spinningFriction=0.05, rollingFriction=0.05)
    #argument preprocessing
    pybullet_instance.changeDynamics(obj.id, -1, mass=.03, restitution=.95, lateralFriction=1)
    arg_dict = args.copy()
    if args['action'] == 'Joint Velocity':
        arg_dict['ik_flag'] = False
    else:
        arg_dict['ik_flag'] = True

    # replay buffer
    replay_buffer = ReplayBufferPriority(buffer_size=4080000)
    
    # environment and recording
    hand_type = this_hand.split('/')[0]
    env = multiprocess_env.MultiprocessSingleShapeEnv(pybullet_instance, hand=hand, obj=obj, hand_type=hand_type, rand_start=args['rstart'])

    # env = rl_env.ExpertEnv(hand=hand, obj=cylinder)
    
    # Create phase
    manipulation = multiprocess_manipulation_phase.MultiprocessManipulation(
        hand, obj, x, y, state, action, reward, env, replay_buffer=replay_buffer, args=arg_dict)
    
    
    # data recording
    record_data = MultiprocessRecordData(rank,
        data_path=args['save_path'], state=state, action=action, reward=reward, save_all=False, controller=manipulation.controller)
    
    
    gym_env = multiprocess_gym_wrapper.MultiprocessGymWrapper(env, manipulation, record_data, args)
    return gym_env, args, [pose_list,eval_pose_list]



def main(filepath = None):
    num_cpu = multiprocessing.cpu_count() # Number of processes to use
    # Create the vectorized environment
    if filepath is None:
        filename = 'FTP_halfstate_A_rand'
        filepath = './data/' + filename +'/experiment_config.json'
        thing = 'run'
    else:
        thing = 'run'

   
    with open(filepath, 'r') as argfile:
        args = json.load(argfile)
    if args['model'] == 'PPO':
        model_type = PPO
    elif 'DDPG' in args['model']:
        model_type = DDPG
    elif 'TD3' in args['model']:
        model_type = TD3
    if thing == 'eval':
        print('LOADING A MODEL')
        import pybullet as p2
        eval_env , _, poses= make_pybullet(filepath,p2, [1,16])
        eval_env.evaluate()
        model = model_type("MlpPolicy", eval_env, tensorboard_log=args['tname'], policy_kwargs={'log_std_init':-2.3}).load(args['load_path']+'best_model', env=eval_env)
        for _ in range(1200):
            obs = eval_env.reset()
            done = False
            # print(np.shape(obs))
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, _ = eval_env.step(action)
        return
    vec_env = SubprocVecEnv([make_env(filepath,[i,num_cpu]) for i in range(num_cpu)])


    # import pybullet as p2
    # eval_env, args, points = make_pybullet(filepath,p2, 100)
    # this_path = os.path.abspath(__file__)
    # overall_path = os.path.dirname(os.path.dirname(os.path.dirname(this_path)))

    train_timesteps = int(args['evaluate']*(args['tsteps']+1)/num_cpu)
    callback = multiprocess_gym_wrapper.MultiEvaluateCallback(vec_env,n_eval_episodes=int(1200), eval_freq=train_timesteps, best_model_save_path=args['save_path'])
    # Stable Baselines provides you with make_vec_env() helper
    # which does exactly the previous steps for you.
    # You can choose between `DummyVecEnv` (usually faster) and `SubprocVecEnv`
    # env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=SubprocVecEnv)
    # wandb.init(project = 'StableBaselinesWandBTest')
    
    if thing == 'transfer':
        model = model_type("MlpPolicy", vec_env, tensorboard_log=args['tname'], policy_kwargs={'log_std_init':-2.3}).load(args['load_path']+'best_model', env=vec_env)
        print('LOADING A MODEL')
    elif thing == 'run':
        model = model_type("MlpPolicy", vec_env,tensorboard_log=args['tname'])

    model.learn(total_timesteps=args['epochs']*(args['tsteps']+1), callback=callback)
    # model.save('./data/'+filename+'/best_policy')
    vec_env.env_method('evaluate')
    for _ in range(1200):
        obs =  vec_env.env_method('reset')
        done = False
        print(np.shape(obs))
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = vec_env[0].step(action)

if __name__ == '__main__':
    main()