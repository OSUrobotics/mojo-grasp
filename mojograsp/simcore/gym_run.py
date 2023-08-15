#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 10:53:58 2023

@author: orochi
"""

import pybullet as p
import pybullet_data
from demos.rl_demo import rl_env
from demos.rl_demo import manipulation_phase_rl
# import rl_env
from demos.rl_demo.rl_state import StateRL, GoalHolder, RandomGoalHolder
from demos.rl_demo import rl_action
from demos.rl_demo import rl_reward
from demos.rl_demo import rl_gym_wrapper
import pandas as pd
from mojograsp.simcore.record_data import RecordDataJSON, RecordDataPKL,  RecordDataRLPKL
from mojograsp.simobjects.two_finger_gripper import TwoFingerGripper
from mojograsp.simobjects.ik_gripper import IKGripper
from mojograsp.simobjects.object_with_velocity import ObjectWithVelocity
from mojograsp.simcore.priority_replay_buffer import ReplayBufferPriority
from mojograsp.simcore.data_combination import data_processor
import pickle as pkl
import json
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import wandb
import numpy as np
import os
import time
# from stable_baselines3.DQN import MlpPolicy
# from stable_baselines3.common.cmd_util import make_vec_env

def run_pybullet(filepath, window=None, runtype='run', episode_number=None, action_list = None):
    # resource paths
    
    with open(filepath, 'r') as argfile:
        args = json.load(argfile)
    
    if runtype =='run':
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
            df2 = pd.read_csv('/home/mothra/mojo-grasp/demos/rl_demo/resources/test_points.csv', index_col=False)
            xeval = [0.045, 0, -0.045, -0.06, -0.045, 0, 0.045, 0.06]
            yeval = [-0.045, -0.06, -0.045, 0, 0.045, 0.06, 0.045, 0]
            eval_names = ['SE','S','SW','W','NW','N','NE','E'] 
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
            xeval = [0.045, 0, -0.045, -0.06, -0.045, 0, 0.045, 0.06]
            yeval = [-0.045, -0.06, -0.045, 0, 0.045, 0.06, 0.045, 0]
            eval_names = ['SE','S','SW','W','NW','N','NE','E'] 

    elif runtype=='eval':
        # df = pd.read_csv('/home/orochi/mojo/mojo-grasp/demos/rl_demo/resources/test_points.csv', index_col=False)
        df = pd.read_csv(args['points_path'], index_col=False)
        print('EVALUATING BOOOIIII')
        x = df["x"]
        y = df["y"]
        xeval = x
        yeval = y
        # xeval = [0.045, 0, -0.045, -0.06, -0.045, 0, 0.045, 0.06]
        # yeval = [-0.045, -0.06, -0.045, 0, 0.045, 0.06, 0.045, 0]
        eval_names = ['eval']*500 
    elif runtype=='replay':
        df = pd.read_csv(args['points_path'], index_col=False)
        x = df["x"]
        y = df["y"]
        xeval = [0.045, 0, -0.045, -0.06, -0.045, 0, 0.045, 0.06]
        yeval = [-0.045, -0.06, -0.045, 0, 0.045, 0.06, 0.045, 0]
        eval_names = ['SE','S','SW','W','NW','N','NE','E'] 
        if action_list == None:
            with open('/home/mothra/mojo-grasp/demos/rl_demo/data/ftp_friction_fuckery/Train/episode_99977.pkl','rb') as fol:
                data = pkl.load(fol)
            action_list = data#np.array(data)
    names = ['AsteriskSE.pkl','AsteriskS.pkl','AsteriskSW.pkl','AsteriskW.pkl','AsteriskNW.pkl','AsteriskN.pkl','AsteriskNE.pkl','AsteriskE.pkl']
    pose_list = [[i,j] for i,j in zip(x,y)]
    eval_pose_list = [[i,j] for i,j in zip(xeval,yeval)]
    print(args)
    try:
        if (args['viz']) | (runtype=='replay'):
            physics_client = p.connect(p.GUI)
        else:
            physics_client = p.connect(p.DIRECT)
    except KeyError:
        physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -10)
    p.setPhysicsEngineParameter(contactBreakingThreshold=.001)
    p.resetDebugVisualizerCamera(cameraDistance=.02, cameraYaw=0, cameraPitch=-89.9999,
                                 cameraTargetPosition=[0, 0.1, 0.5])
    
    # load objects into pybullet
    plane_id = p.loadURDF("plane.urdf", flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
    hand_id = p.loadURDF(args['hand_path'], useFixedBase=True,
                         basePosition=[0.0, 0.0, 0.05], flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
    obj_id = p.loadURDF(args['object_path'], basePosition=[0.0, 0.10, .05], flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
    # print('expected inertia  ', [0.000029435425,0.000029435425,0.00000725805])
    # print('cube dynamics info',p.getDynamicsInfo(obj_id,-1))
    # Create TwoFingerGripper Object and set the initial joint positions
    # hand = TwoFingerGripper(hand_id, path=args['hand_path'])

    hand = IKGripper(hand_id, path=args['hand_path'])
    
    # p.resetJointState(hand_id, 0, -0.4)
    # p.resetJointState(hand_id, 1, 1.2)
    # p.resetJointState(hand_id, 3, 0.4)
    # p.resetJointState(hand_id, 4, -1.2)
    
    # p.resetJointState(hand_id, 0, 0)
    # p.resetJointState(hand_id, 1, 0)
    # p.resetJointState(hand_id, 3, 0)
    # p.resetJointState(hand_id, 4, 0)
    # change visual of gripper
    p.changeVisualShape(hand_id, -1, rgbaColor=[0.3, 0.3, 0.3, 1])
    p.changeVisualShape(hand_id, 0, rgbaColor=[1, 0.5, 0, 1])
    p.changeVisualShape(hand_id, 1, rgbaColor=[0.3, 0.3, 0.3, 1])
    p.changeVisualShape(hand_id, 3, rgbaColor=[1, 0.5, 0, 1])
    p.changeVisualShape(hand_id, 4, rgbaColor=[0.3, 0.3, 0.3, 1])
    # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
    # p.setTimeStep(1/2400)
    obj = ObjectWithVelocity(obj_id, path=args['object_path'],name='obj_2')
    # p.addUserDebugPoints([[0.2,0.1,0.0],[1,0,0]],[[1,0.0,0],[0.5,0.5,0.5]], 1)
    # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    
    # For standard loaded goal poses
    if args['task'] == 'unplanned_random':
        goal_poses = RandomGoalHolder([0.02,0.065])
    else:    
        goal_poses = GoalHolder(pose_list)

    eval_goal_poses = GoalHolder(eval_pose_list,eval_names)
    # time.sleep(10)
    # state, action and reward
    state = StateRL(objects=[hand, obj, goal_poses], prev_len=args['pv'],eval_goals = eval_goal_poses)
    
    if args['freq'] ==240:
        action = rl_action.ExpertAction()
    else:
        action = rl_action.InterpAction(args['freq'])
    
    reward = rl_reward.ExpertReward()
    p.changeDynamics(plane_id,-1,lateralFriction=0.05, spinningFriction=0.05, rollingFriction=0.05)
    #argument preprocessing
    p.changeDynamics(obj.id, -1, mass=.03, restitution=.95, lateralFriction=1, localInertiaDiagonal=[0.000029435425,0.000029435425,0.00000725805])

    arg_dict = args.copy()
    if args['action'] == 'Joint Velocity':
        arg_dict['ik_flag'] = False
    else:
        arg_dict['ik_flag'] = True

    # replay buffer
    replay_buffer = ReplayBufferPriority(buffer_size=4080000)
    
    # environment and recording
    env = rl_env.ExpertEnv(hand=hand, obj=obj, hand_type=arg_dict['hand'], rand_start=args['rstart'])

    # env = rl_env.ExpertEnv(hand=hand, obj=cylinder)
    
    # Create phase
    manipulation = manipulation_phase_rl.ManipulationRL(
        hand, obj, x, y, state, action, reward, replay_buffer=replay_buffer, args=arg_dict)
    
    
    # data recording
    record_data = RecordDataRLPKL(
        data_path=args['save_path'], state=state, action=action, reward=reward, save_all=False, controller=manipulation.controller)
    
    
    gym_env = rl_gym_wrapper.GymWrapper(env, manipulation, record_data, args)
    train_timesteps = args['evaluate']*151
    callback = rl_gym_wrapper.EvaluateCallback(gym_env,n_eval_episodes=8, eval_freq=train_timesteps, best_model_save_path=args['save_path'])
    # gym_env = Monitor(gym_env,args['save_path']+'Test/')
    # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    # p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    # check_env(gym_env, warn=True)
    if 'entropy' in args.keys():
        ent = args['entropy']
    else:
        ent = 0.0
    if runtype == 'run':
        wandb.init(project = 'StableBaselinesWandBTest')

        model = PPO("MlpPolicy", gym_env, tensorboard_log=args['tname'], policy_kwargs={'log_std_init':-1}, ent_coef=ent)
        # gym_env = make_vec_env(lambda: gym_env, n_envs=1)
        gym_env.train()
        model.learn(args['epochs']*151, callback=callback)
        d = data_processor(args['save_path'] + 'Train/')
        d.load_limited()
        d.save_all()

    elif runtype == 'eval':
        model = PPO("MlpPolicy", gym_env, tensorboard_log=args['tname'], policy_kwargs={'log_std_init':-1}).load(args['save_path']+'best_model')
        gym_env.evaluate()
        # obj_pos = [0.0, 0.1, 0.05]
        # joint_angs = [ -.85,1.3,0.85,-1.3]
            # p.resetJointState(hand_id, 0, -.725)
            # p.resetJointState(hand_id, 1, 1.45)
            # p.resetJointState(hand_id, 3, .725)
            # p.resetJointState(hand_id, 4, -1.45)
        for _ in range(500):
            obs = gym_env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                # print("Step {}".format(step + 1))
                # print("Action: ", action, type(action))
                mirrored_action = np.array([-action[2], action[3],-action[0],action[1]])
                # print('mirrored action: ', mirrored_action)
                obs, reward, done, info = gym_env.step(action)
                # print('obs=', obs, 'reward=', reward, 'done=', done)

    elif runtype == 'cont':
        pass
    
    elif runtype == 'transfer':
        pass
    
    elif runtype == 'replay':
        gym_env.evaluate()
        actions = []
        print(action_list['timestep_list'])
        actions = [a['action']['actor_output'] for a in action_list['timestep_list']]
            
        for _ in range(1):
            obs = gym_env.reset()
            for step in range(151):
                # action, _ = model.predict(obs, deterministic=True)
                print(action_list)
                action = actions[step]
                # print("Step {}".format(step + 1))
                # print("Action: ", action, type(action))
                # mirrored_action = np.array([-action[2], action[3],-action[0],action[1]])
                obs, reward, done, info = gym_env.step(np.array(action),viz=True)
                # print('obs=', obs, 'reward=', reward, 'done=', done)
                # env.render(mode='console')

def main():
    this_path = os.path.abspath(__file__)
    overall_path = os.path.dirname(os.path.dirname(os.path.dirname(this_path)))
    # run_pybullet(overall_path+'/demos/rl_demo/data/ftp_friction_fuckery/experiment_config.json', runtype='replay')

    run_pybullet(overall_path+'/demos/rl_demo/data/ja_monitor_benchmark/experiment_config.json', runtype='run')

    # run_pybullet(overall_path+'/demos/rl_demo/data/ja_monitor_benchmark/experiment_config.json',runtype='eval')

# DO A REPLAY OF JA-testing episode 99924, 99918
if __name__ == '__main__':
    main()
