#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 10:53:58 2023

@author: orochi
"""

import pybullet as p
from pybullet_utils import bullet_client as bc
import pybullet_data
from demos.rl_demo import rl_env
from demos.rl_demo import manipulation_phase_rl
# import rl_env

from demos.rl_demo import multiprocess_state as rl_state
from demos.rl_demo.rl_state import GoalHolder, RandomGoalHolder

from demos.rl_demo import rl_action
from demos.rl_demo import multiprocess_reward as rl_reward
from demos.rl_demo import rl_gym_wrapper

import pandas as pd

from mojograsp.simcore.record_data import RecordDataJSON, RecordDataPKL,  RecordDataRLPKL
from mojograsp.simobjects.two_finger_gripper import TwoFingerGripper
from mojograsp.simobjects.object_with_velocity import ObjectWithVelocity
from mojograsp.simcore.priority_replay_buffer import ReplayBufferPriority
import pickle as pkl
import json
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import os
import numpy as np
from typing import Callable
from mojograsp.simcore.data_combination import data_processor

# from stable_baselines3.DQN import MlpPolicy


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

def make_env(env, manipulation, record_data, args):
    def _init():
        gym_env = rl_gym_wrapper.GymWrapper(env, manipulation, record_data, args)
        return gym_env
    return _init

def run_pybullet(filepath, window=None, runtype='run', episode_number=None):
    # resource paths
    this_path = os.path.abspath(__file__)
    overall_path = os.path.dirname(os.path.dirname(os.path.dirname(this_path)))
    with open(filepath, 'r') as argfile:
        args = json.load(argfile)
    
    if (runtype =='run') | (runtype =='transfer'):
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
        elif 'big_random' == args['task']:
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

    elif runtype=='eval':
        if args['task'] == 'forward':
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
    np.random.shuffle(pose_list)
    eval_pose_list = [[i,j] for i,j in zip(xeval,yeval)]
    print(args)
    
    envs = []
    manipulation_phases = []
    data_recorders = []
    for _ in range(2):
        # physics_client = b1.connect(p.DIRECT)
        b1 = bc.BulletClient(p.DIRECT)
        # print('connected to physics client',physics_client)
        b1.setAdditionalSearchPath(pybullet_data.getDataPath())
        b1.setGravity(0, 0, -10)
        b1.setPhysicsEngineParameter(contactBreakingThreshold=.001)
        b1.resetDebugVisualizerCamera(cameraDistance=.02, cameraYaw=0, cameraPitch=-89.9999,
                                     cameraTargetPosition=[0, 0.1, 0.5])
        
        # load objects into pybullet
        plane_id = b1.loadURDF("plane.urdf", flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
        hand_id = b1.loadURDF(args['hand_path'], useFixedBase=True,
                             basePosition=[0.0, 0.0, 0.05], flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
        obj_id = b1.loadURDF(args['object_path'], basePosition=[0.0, 0.10, .05], flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
        
        # Create TwoFingerGripper Object and set the initial joint positions
        hand = TwoFingerGripper(hand_id, path=args['hand_path'],physicsClientId=b1)

        # change visual of gripper
        b1.changeVisualShape(hand_id, -1, rgbaColor=[0.3, 0.3, 0.3, 1])
        b1.changeVisualShape(hand_id, 0, rgbaColor=[1, 0.5, 0, 1])
        b1.changeVisualShape(hand_id, 1, rgbaColor=[0.3, 0.3, 0.3, 1])
        b1.changeVisualShape(hand_id, 3, rgbaColor=[1, 0.5, 0, 1])
        b1.changeVisualShape(hand_id, 4, rgbaColor=[0.3, 0.3, 0.3, 1])

        obj = ObjectWithVelocity(obj_id, path=args['object_path'],name='obj_2',physicsClientId=b1)
        
        # For standard loaded goal poses
        if args['task'] == 'unplanned_random':
            goal_poses = RandomGoalHolder([0.02,0.065])
        else:    
            goal_poses = GoalHolder(pose_list)
        
        # For randomized poses
        try:
            eval_goal_poses = GoalHolder(eval_pose_list,eval_names)
        except NameError:
            print('No names')
            eval_goal_poses = GoalHolder(eval_pose_list)
        # time.sleep(10)
        # state, action and reward
        state = rl_state.MultiprocessState(objects=[hand, obj, goal_poses], prev_len=args['pv'],eval_goals = eval_goal_poses, physicsClientId=b1)
        
        if args['freq'] ==240:
            action = rl_action.ExpertAction()
        else:
            action = rl_action.InterpAction(args['freq'])
        
        reward = rl_reward.MultiprocessReward(b1)
        b1.changeDynamics(plane_id,-1,lateralFriction=0.05, spinningFriction=0.05, rollingFriction=0.05)
        #argument preprocessing
        b1.changeDynamics(obj.id, -1, mass=.03, restitution=.95, lateralFriction=1, localInertiaDiagonal=[0.000029435425,0.000029435425,0.00000725805])
        arg_dict = args.copy()
        if args['action'] == 'Joint Velocity':
            arg_dict['ik_flag'] = False
        else:
            arg_dict['ik_flag'] = True
    
        # replay buffer
        replay_buffer = ReplayBufferPriority(buffer_size=4080000)
        
        # environment and recording
        env = rl_env.ExpertEnv(hand=hand, obj=obj, hand_type=arg_dict['hand'], rand_start=args['rstart'],physicsClientId=b1)
    
        # env = rl_env.ExpertEnv(hand=hand, obj=cylinder)
        # Create phase
        manipulation = manipulation_phase_rl.ManipulationRL(
            hand, obj, x, y, state, action, reward,env, replay_buffer=replay_buffer, args=arg_dict,physicsClientId=b1)
        
        
        # data recording
        record_data = RecordDataRLPKL(
            data_path=args['save_path'], state=state, action=action, reward=reward, save_all=False, controller=manipulation.controller)
        
        
        # gym_env = rl_gym_wrapper.GymWrapper(env, manipulation, record_data, args)
        envs.append(env)
        manipulation_phases.append(manipulation)
        data_recorders.append(record_data)

    env = SubprocVecEnv([make_env(envs[i], manipulation_phases[i], data_recorders[i], arg_dict) for i in range(2)])

    # b1.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    # b1.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    # check_env(gym_env, warn=True)
    if 'entropy' in args.keys():
        ent = args['entropy']
    else:
        ent = 0.0
    if runtype == 'run':
        # best_performance = -1000
        model = PPO("MlpPolicy", env, tensorboard_log=args['tname'], policy_kwargs={'log_std_init':-2.3}, ent_coef=ent,learning_rate=linear_schedule(args['learning_rate']))
        # gym_env = make_vec_env(lambda: gym_env, n_envs=1)
        # train_timesteps = args['evaluate']*151
        env.train()
        model.learn(args['epochs']*(args['tsteps']+1))
        # for epoch in range(int(args['epochs']/args['evaluate'])):
        #     for gym_env in env:
        #         gym_env.train()
        #     model.learn(train_timesteps)
        #     for gym_env in env:
        #         gym_env.evaluate()
        #     performance, _ = evaluate_policy(model, env,n_eval_episodes=8)
        #     print('done evaluating')
        #     if performance > best_performance:
        #         best_performance = performance
        #         model.save(args['save_path']+'policy')
        #         temp = model.get_parameters()
        #         with open(args['save_path']+'parameters.pkl','wb') as file:
        #             pkl.dump(temp,file)
        # d = data_processor(args['save_path'] + 'Train/')
        # d.load_data()
        # d.save_all()
        d = data_processor(args['save_path'] + 'Train/')
        d.load_limited()
        d.save_all()
        model.save(args['save_path']+'best_model')
        env.evaluate()
        env.episode_type = 'eval'
        for _ in range(len(eval_goal_poses)):
            obs = env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)

def main():
    run_pybullet('/home/orochi/mojo/mojo-grasp/demos/rl_demo/data/multiprocess_test/experiment_config.json',runtype='run')
    # run_pybullet('/home/orochi/mojo/mojo-grasp/demos/rl_demo/data/ftp_experiment/experiment_config.json',runtype='eval')
if __name__ == '__main__':
    main()