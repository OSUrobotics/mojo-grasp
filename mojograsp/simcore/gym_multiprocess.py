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
from demos.rl_demo.rl_state import StateRL, GoalHolder, RandomGoalHolder
from demos.rl_demo import rl_action
from demos.rl_demo import rl_reward
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
import wandb

# from stable_baselines3.DQN import MlpPolicy


def make_env(env, manipulation, record_data, args):
    def _init():
        gym_env = rl_gym_wrapper.GymWrapper(env, manipulation, record_data, args)
        return gym_env
    return _init

def run_pybullet(filepath, window=None, runtype='run', episode_number=None):
    # resource paths
    wandb.init(project = 'StableBaselinesWandBTest')
    with open(filepath, 'r') as argfile:
        args = json.load(argfile)
    
    if runtype =='run':
        if args['task'] == 'asterisk':
            x = [0.03, 0, -0.03, -0.04, -0.03, 0, 0.03, 0.04]
            y = [-0.03, -0.04, -0.03, 0, 0.03, 0.04, 0.03, 0]
            xeval = x
            yeval = y
            
        elif 'random' == args['task']:
            df = pd.read_csv(args['points_path'], index_col=False)
            x = df["x"]
            y = df["y"]
            df2 = pd.read_csv('/home/orochi/mojo/mojo-grasp/demos/rl_demo/resources/test_points.csv', index_col=False)
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
    elif runtype=='eval':
        df = pd.read_csv('/home/orochi/mojo/mojo-grasp/demos/rl_demo/resources/test_points.csv', index_col=False)
        print('EVALUATING BOOOIIII')
        x = df["x"]
        y = df["y"]
        xeval = x
        yeval = y
        xeval = [0.045, 0, -0.045, -0.06, -0.045, 0, 0.045, 0.06]
        yeval = [-0.045, -0.06, -0.045, 0, 0.045, 0.06, 0.045, 0]
        eval_names = ['SE','S','SW','W','NW','N','NE','E'] 
    elif runtype=='replay':
        df = pd.read_csv(args['points_path'], index_col=False)
        x = df["x"]
        y = df["y"]
        df2 = pd.read_csv('/home/orochi/mojo/mojo-grasp/demos/rl_demo/resources/test_points.csv', index_col=False)
        xeval = df2["x"]
        yeval = df2["y"]
    
    names = ['AsteriskSE.pkl','AsteriskS.pkl','AsteriskSW.pkl','AsteriskW.pkl','AsteriskNW.pkl','AsteriskN.pkl','AsteriskNE.pkl','AsteriskE.pkl']
    pose_list = [[i,j] for i,j in zip(x,y)]
    eval_pose_list = [[i,j] for i,j in zip(xeval,yeval)]
    print(args)
    
    envs = []
    manipulation_phases = []
    data_recorders = []
    for _ in range(4):
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
        
        # b1.resetJointState(hand_id, 0, -0.4)
        # b1.resetJointState(hand_id, 1, 1.2)
        # b1.resetJointState(hand_id, 3, 0.4)
        # b1.resetJointState(hand_id, 4, -1.2)
        
        # b1.resetJointState(hand_id, 0, 0)
        # b1.resetJointState(hand_id, 1, 0)
        # b1.resetJointState(hand_id, 3, 0)
        # b1.resetJointState(hand_id, 4, 0)
        # change visual of gripper
        b1.changeVisualShape(hand_id, -1, rgbaColor=[0.3, 0.3, 0.3, 1])
        b1.changeVisualShape(hand_id, 0, rgbaColor=[1, 0.5, 0, 1])
        b1.changeVisualShape(hand_id, 1, rgbaColor=[0.3, 0.3, 0.3, 1])
        b1.changeVisualShape(hand_id, 3, rgbaColor=[1, 0.5, 0, 1])
        b1.changeVisualShape(hand_id, 4, rgbaColor=[0.3, 0.3, 0.3, 1])
        # b1.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
        # b1.setTimeStep(1/2400)
        obj = ObjectWithVelocity(obj_id, path=args['object_path'],name='obj_2',physicsClientId=b1)
        # b1.addUserDebugPoints([[0.2,0.1,0.0],[1,0,0]],[[1,0.0,0],[0.5,0.5,0.5]], 1)
        # b1.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        
        # For standard loaded goal poses
        if args['task'] == 'unplanned_random':
            goal_poses = RandomGoalHolder([0.02,0.065])
        else:    
            goal_poses = GoalHolder(pose_list)
        
        # For randomized poses
        
        
        eval_goal_poses = GoalHolder(eval_pose_list,eval_names)
        # time.sleep(10)
        # state, action and reward
        state = StateRL(objects=[hand, obj, goal_poses], prev_len=args['pv'],eval_goals = eval_goal_poses, physicsClientId=b1)
        
        if args['freq'] ==240:
            action = rl_action.ExpertAction()
        else:
            action = rl_action.InterpAction(args['freq'])
        
        reward = rl_reward.ExpertReward(b1)
        b1.changeDynamics(plane_id,-1,lateralFriction=0.05, spinningFriction=0.05, rollingFriction=0.05)
        #argument preprocessing
        b1.changeDynamics(obj.id, -1, mass=.03, restitution=.95, lateralFriction=1)
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
            hand, obj, x, y, state, action, reward, replay_buffer=replay_buffer, args=arg_dict,physicsClientId=b1)
        
        
        # data recording
        record_data = RecordDataRLPKL(
            data_path=args['save_path'], state=state, action=action, reward=reward, save_all=False, controller=manipulation.controller)
        
        
        # gym_env = rl_gym_wrapper.GymWrapper(env, manipulation, record_data, args)
        envs.append(env)
        manipulation_phases.append(manipulation)
        data_recorders.append(record_data)

    env = SubprocVecEnv([make_env(envs[i], manipulation_phases[i], data_recorders[i], arg_dict) for i in range(4)])
    # b1.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    # b1.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    # check_env(gym_env, warn=True)
    if runtype == 'run':
        # best_performance = -1000
        model = PPO("MlpPolicy", env, tensorboard_log=args['tname'], policy_kwargs={'log_std_init':-1})
        # gym_env = make_vec_env(lambda: gym_env, n_envs=1)
        # train_timesteps = args['evaluate']*151
        model.learn(args['epochs']*151)
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
        env.eval = True
        env.eval_names = names
        

        for _ in range(8):
            obs = env.reset()
            for step in range(151):
                action, _ = model.predict(obs, deterministic=True)
                # print("Step {}".format(step + 1))
                # print("Action: ", action)
                obs, reward, done, info = env.step(action)
                # print('obs=', obs, 'reward=', reward, 'done=', done)
                # env.render(mode='console')

def main():
    run_pybullet('/home/orochi/mojo/mojo-grasp/demos/rl_demo/data/ftp_experiment/experiment_config.json',runtype='run')
    # run_pybullet('/home/orochi/mojo/mojo-grasp/demos/rl_demo/data/ftp_experiment/experiment_config.json',runtype='eval')
if __name__ == '__main__':
    main()