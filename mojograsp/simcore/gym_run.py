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
from demos.rl_demo.rl_state import StateRL
from mojograsp.simcore.goal_holder import  GoalHolder, RandomGoalHolder
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
from stable_baselines3 import A2C, PPO, TD3, HerReplayBuffer
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.s
import wandb
import numpy as np
import os
import time
from typing import Callable
# from stable_baselines3.DQN import MlpPolicy
# from stable_baselines3.common.cmd_util import make_vec_env
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

def run_pybullet(filepath, runtype='run', episode_number=None, action_list = None, hand = None):
    # resource paths
    this_path = os.path.abspath(__file__)
    overall_path = os.path.dirname(os.path.dirname(os.path.dirname(this_path)))
    
    with open(filepath, 'r') as argfile:
        args = json.load(argfile)
    args['hand_file_list'] =["2v2_50.50_50.50_1.1_53/hand/2v2_50.50_50.50_1.1_53.urdf"]
    # args['rstart'] = 'no'
    if hand is not None:
        args['hand'] = hand
        # args['rstart'] = 'none'
        if hand=='2v2-B':
            args['hand_path']="/home/mothra/mojo-grasp/demos/rl_demo/resources/2v2_Hand_B/hand/2v2_65.35_65.35_1.1_53.urdf"
            print('shit')
        elif hand=='2v2':
            args['hand_path']="/home/mothra/mojo-grasp/demos/rl_demo/resources/2v2_Hand_A/hand/2v2_50.50_50.50_1.1_53.urdf"

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
        elif args['task'] == 'Rotation':
            x =[0]*500
            y = [0]*500
            theta = np.random.uniform(-np.pi,np.pi,500)
            eval_theta = np.random.uniform(-np.pi,np.pi,500)
    elif runtype=='eval':
        if (args['task'] == 'big_random') |(args['task']=='random') | (args['task']=='multi'):
            print('this is the one ')
            df = pd.read_csv(args['points_path'], index_col=False)
            x = df["x"]
            y = df["y"]
            df2 = pd.read_csv(overall_path + '/demos/rl_demo/resources/test_points_big.csv', index_col=False)
            xeval = df2['x']
            yeval = df2['y']
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
    
    elif runtype=='replay':

        df = pd.read_csv(args['points_path'], index_col=False)
        x = df["x"]
        y = df["y"]
        xeval = [0.045, 0, -0.045, -0.06, -0.045, 0, 0.045, 0.06]
        yeval = [-0.045, -0.06, -0.045, 0, 0.045, 0.06, 0.045, 0]
        eval_names = ['SE','S','SW','W','NW','N','NE','E'] 
        if action_list == None:
            with open(overall_path + '/demos/rl_demo/data/FTP_halfstate_A_rand/Eval_A/Evaluate_215.pkl','rb') as fol:
                data = pkl.load(fol)
            action_list = data#np.array(data)
    names = ['AsteriskSE.pkl','AsteriskS.pkl','AsteriskSW.pkl','AsteriskW.pkl','AsteriskNW.pkl','AsteriskN.pkl','AsteriskNE.pkl','AsteriskE.pkl']
    pose_list = [[i,j] for i,j in zip(x,y)]

    eval_pose_list = [[0,0.07],[0.0495,0.0495],[0.07,0.0],[0.0495,-0.0495],[-0.0495,-0.0495],[-0.07,0.0],[-0.0495,0.0495]]
    eval_pose_list = [[i,j] for i,j in zip(xeval,yeval)]
    if args['task'] =='Rotation':
        pose_list = [[i,j,k] for i,j,k in zip(x,y,theta)]
        eval_pose_list = [[i,j,k] for i,j,k in zip(x,y,eval_theta)]
    np.random.shuffle(pose_list)
    
    print(args)
    try:
        if (args['viz']) | (runtype=='replay') | (runtype=='eval'):
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
    if len(args['hand_file_list']) > 1:
        print('multiple hands not supported for gym_run.py, try with multiprocess_gym_run.py instead')
        return
    # print(args['hand_path']+'/'+args['hand_file_list'][0])
    # print("loading hand",args['hand_path'] +'/'+ args['hand_file_list'][0])
    hand_id = p.loadURDF(args['hand_path'], useFixedBase=True,
                         basePosition=[0.0, 0.0, 0.05], flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
    # hand_id = p.loadURDF(args['hand_path']+'/'+args['hand_file_list'][0], useFixedBase=True,
    #                      basePosition=[0.0, 0.0, 0.05], flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
    obj_id = p.loadURDF(args['object_path'], basePosition=[0.0, 0.10, .05], flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)

    # Create TwoFingerGripper Object and set the initial joint positions
    # hand = TwoFingerGripper(hand_id, path=args['hand_path'])


    # key_file = '/home/mothra/mojo-grasp/demos/rl_demo'
    # key_file = os.path.join(key_file,'resources','hand_bank','hand_params.json')
    # with open(key_file,'r') as hand_file:
    #     hand_info = json.load(hand_file)
    # this_hand = args['hand_file_list'][0]
    # hand_type = this_hand.split('/')[0]
    # print(hand_type)
    # hand_keys = hand_type.split('_')
    # info_1 = hand_info[hand_keys[-1]][hand_keys[1]]
    # info_2 = hand_info[hand_keys[-1]][hand_keys[2]]
    # hand_param_dict = {"link_lengths":[info_1['link_lengths'],info_2['link_lengths']],
    #                    "starting_angles":[info_1['start_angles'][0],info_1['start_angles'][1],-info_2['start_angles'][0],-info_2['start_angles'][1]],
    #                    "palm_width":info_1['palm_width'],
    #                    "hand_name":hand_type}
    sa = [-0.725,1.45]
    hand_param_dict = {"link_lengths":[[[0, 0.0936, 0], [0, 0.0504, 0]],[[0, 0.0936, 0], [0, 0.0504, 0]]],
                       "starting_angles":[sa[0],sa[1],-sa[0],-sa[1]],
                       "palm_width":0.053,
                       "hand_name":'B'}
    hand = TwoFingerGripper(hand_id, path=args['hand_path'],hand_params=hand_param_dict)

    # hand = IKGripper(hand_id, path=args['hand_path']+'/'+args['hand_file_list'][0])
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
        goal_poses = RandomGoalHolder([0.0025,0.065])
    else:    
        goal_poses = GoalHolder(pose_list)

    try:
        eval_goal_poses = GoalHolder(eval_pose_list,eval_names)
    except NameError:
        print('No names')
        eval_goal_poses = GoalHolder(eval_pose_list)
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
    # env = rl_env.ExpertEnv(hand=hand, obj=obj, hand_type=arg_dict['hand'], rand_start=args['rstart'])
    hand_type = arg_dict['hand_file_list'][0].split('/')[0]
    env = rl_env.SingleShapeEnv(hand=hand, obj=obj, hand_type=hand_type, rand_start=args['rstart'])
    # env = rl_env.ExpertEnv(hand=hand, obj=cylinder)
    
    # Create phase
    manipulation = manipulation_phase_rl.ManipulationRL(
        hand, obj, x, y, state, action, reward, env, replay_buffer=replay_buffer, args=arg_dict)
    
    
    # data recording
    record_data = RecordDataRLPKL(
        data_path=args['save_path'], state=state, action=action, reward=reward, save_all=False, controller=manipulation.controller)
    
    
    gym_env = rl_gym_wrapper.GymWrapper(env, manipulation, record_data, args)
    train_timesteps = args['evaluate']*(args['tsteps']+1)
    callback = rl_gym_wrapper.EvaluateCallback(gym_env,n_eval_episodes=len(eval_pose_list), eval_freq=train_timesteps, best_model_save_path=args['save_path'])
    # gym_env = Monitor(gym_env,args['save_path']+'Test/')
    # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    # p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    # check_env(gym_env, warn=True)
    if 'entropy' in args.keys():
        ent = args['entropy']
    else:
        ent = 0.0
    if runtype == 'run':
        # wandb.init(project = 'StableBaselinesWandBTest')
        
        model = PPO("MlpPolicy", gym_env, tensorboard_log=args['tname'], policy_kwargs={'log_std_init':-2.3}, ent_coef=ent,learning_rate=linear_schedule(args['learning_rate']))

        # gym_env = make_vec_env(lambda: gym_env, n_envs=1)
        gym_env.train()
        model.learn(args['epochs']*(args['tsteps']+1), callback=callback)
        d = data_processor(args['save_path'] + 'Train/')
        d.load_limited()
        d.save_all()
        model.save(args['save_path']+'best_model')
        gym_env.evaluate()
        gym_env.episode_type = 'eval'
        for _ in range(len(eval_goal_poses)):
            obs = gym_env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = gym_env.step(action)

    elif runtype == 'eval':
        model = PPO("MlpPolicy", gym_env, tensorboard_log=args['tname'], policy_kwargs={'log_std_init':-1}).load(args['save_path']+'best_model')
        gym_env.evaluate()

        for _ in range(len(eval_pose_list)):
            obs = gym_env.reset()
            done = False
            step = 0
            while not done:
                # print('step: ',step)
                action, _ = model.predict(obs, deterministic=False)
                print(action)
                # mirrored_action = np.array([-action[2], action[3],-action[0],action[1]])
                obs, reward, done, info = gym_env.step(action, viz=False)
                step +=1

    elif runtype == 'cont':
        pass
    
    elif runtype == 'transfer':
        model = PPO("MlpPolicy", gym_env, tensorboard_log=args['tname'], policy_kwargs={'log_std_init':-2.3}, ent_coef=ent,learning_rate=linear_schedule(args['learning_rate'])).load(args['load_path']+'best_model', env=gym_env)

        gym_env.train()

        model.learn(args['epochs']*(args['tsteps']+1), callback=callback)
        d = data_processor(args['save_path'] + 'Train/')
        d.load_limited()
        d.save_all()
        model.save(args['save_path']+'best_model')
        gym_env.evaluate()
        gym_env.episode_type = 'eval'
        for _ in range(len(eval_goal_poses)):
            obs = gym_env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                
                obs, reward, done, info = gym_env.step(action)
    
    elif runtype == 'replay':
        gym_env.evaluate()
        actions = []
        actions = [a['action']['actor_output'] for a in action_list['timestep_list']]
            
        for _ in range(1):
            obs = gym_env.reset()
            input('start thing?')
            for step in range((args['tsteps']+1)):
                # action, _ = model.predict(obs, deterministic=True)
                # print()
                action = actions[step]
                # print("Step {}".format(step + 1))
                print('step: ',step)
                # temp = np.array([0,0,0,0])
                # print("Action: ", action, type(action))
                # mirrored_action = np.array([-action[2], action[3],-action[0],action[1]])
                obs, reward, done, info = gym_env.step(np.array(action),viz=True)
                # obs, reward, done, info = gym_env.step(np.array(action),viz=True)
                # print('obs=', obs, 'reward=', reward, 'done=', done)
                # time.sleep(0.5)
                # env.render(mode='console')
    p.disconnect()

def main():
    this_path = os.path.abspath(__file__)
    overall_path = os.path.dirname(os.path.dirname(os.path.dirname(this_path)))
    # run_pybullet(overall_path+'/demos/rl_demo/data/single_direction_limited/right/experiment_config.json', runtype='replay', episode_number=9987)
    hand_b_key = "2v2-B"
    hand_a_key = "2v2"
    # run_pybullet(overall_path+'/demos/rl_demo/data/ja_please/experiment_config.json', runtype='run')

    # run_pybullet(overall_path+'/demos/rl_demo/data/FUCK/experiment_config.json', runtype='run')

    # run_pybullet(overall_path+'/demos/rl_demo/data/wedge/wedge_badckward/experiment_config.json',runtype='run')
    file_list = ['forward','forward_right','right','backward_right','backward','backward_left','left','forward_left']
    trimmed_list = ['forward','backward_right','left','forward_left']
    fast_run_list = ['backward_right','backward_left','forward_left']
    double_list = ['f-b', 'l-r', 'diag-up', 'diag-down']
    alt_list = ['f','b']
    # run_pybullet(overall_path+'/demos/rl_demo/data/single_direction_updated_reward/forward/experiment_config.json',runtype='run')
    # run_pybullet(overall_path+'/demos/rl_demo/data/single_direction_updated_reward/backward/experiment_config.json',runtype='run')
    # run_pybullet(overall_path+'/demos/rl_demo/data/single_direction_updated_reward/left/experiment_config.json',runtype='run')
    # run_pybullet(overall_path+'/demos/rl_demo/data/single_direction_updated_reward/right/experiment_config.json',runtype='run')
    # run_pybullet(overall_path+'/demos/rl_demo/data/single_direction_updated_reward/forward_left/experiment_config.json',runtype='run')
    # run_pybullet(overall_path+'/demos/rl_demo/data/single_direction_updated_reward/forward_right/experiment_config.json',runtype='run')
    # run_pybullet(overall_path+'/demos/rl_demo/data/single_direction_updated_reward/backward_left/experiment_config.json',runtype='run')
    # run_pybullet(overall_path + '/demos/rl_demo/data/full_full_ppo/experiment_config.json', runtype='replay', episode_number=50049)
    # for name in double_list:
        # run_pybullet(overall_path + '/demos/rl_demo/data/ftp_her/wedge_' + name + '/experiment_config.json', runtype='run')

    #NOTE WE MAY WANT TO UPDATE THE MAGNITUDE OF THE MAXIMUM MOVEMENT TO BE 1/8 THE SIZE THAT IT WAS TO MATCH THE PREVIOUS SETUP
    # for name in double_list:
    #     run_pybullet(overall_path + '/demos/rl_demo/data/wedge_double/wedge_' + name + '/experiment_config.json', runtype='run')
    # for i in range(1000):
    # run_pybullet(overall_path + '/demos/rl_demo/data/Transfer_to_everything/experiment_config.json', runtype='replay',episode_number=0)
    # run_pybullet(overall_path + '/demos/rl_demo/data/hand_transfer_FTP/experiment_config.json', runtype='eval')
    # run_pybullet(overall_path + '/demos/rl_demo/data/hand_transfer_JA/experiment_config.json', runtype='eval')
    run_pybullet(overall_path + '/demos/rl_demo/data/FTP_halfstate_A_rand/experiment_config.json', runtype='eval', hand=hand_a_key)
    # run_pybullet(overall_path + '/demos/rl_demo/data/FTP_fullstate_A_rand/experiment_config.json', runtype='eval', hand=hand_b_key)
    # run_pybullet(overall_path + '/demos/rl_demo/data/FTP_state_3_old/experiment_config.json', runtype='eval', hand=hand_b_key)
    # run_pybullet(overall_path + '/demos/rl_demo/data/JA_state_3_old/experiment_config.json', runtype='eval', hand=hand_b_key)
    # run_pybullet(overall_path + '/demos/rl_demo/data/JA_fullstate_A_rand/experiment_config.json', runtype='eval', hand=hand_b_key)
    # run_pybullet(overall_path + '/demos/rl_demo/data/JA_halfstate_A_rand/experiment_config.json', runtype='eval', hand=hand_b_key)


    # run_pybullet(overall_path+'/demos/rl_demo/data/JA_state_3_old/experiment_config.json',runtype='replay', episode_number=1)

    # run_pybullet(overall_path+'/demos/rl_demo/data/wedge/wedge_forward_right/experiment_config.json',runtype='run')
    # run_pybullet(overall_path+'/demos/rl_demo/data/wedge/wedge_forward_left/experiment_config.json',runtype='run')
    # run_pybullet(overall_path+'/demos/rl_demo/data/wedge/wedge_left/experiment_config.json',runtype='run')
    # run_pybullet(overall_path+'/demos/rl_demo/data/wedge/wedge_right/experiment_config.json',runtype='run')
    # run_pybullet(overall_path+'/demos/rl_demo/data/wedge/wedge_backward_left/experiment_config.json',runtype='run')
    # run_pybullet(overall_path+'/demos/rl_demo/data/wedge/wedge_backward_right/experiment_config.json',runtype='run')
    # Thoughts: 1. learning to maintain contact within 5k timesteps aka before first evaluation step. might want to tune down contact reward
    # Thoughts: 2. might want to not cap the negative distance reward at -2. It just lets the thing off the hook when it gets sufficiently bad
    # Thoughts: 3. maybe try left and right rather than wedge as a way to slowly expand
if __name__ == '__main__':
    main()
