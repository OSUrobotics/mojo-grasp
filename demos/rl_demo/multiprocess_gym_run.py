#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 10:53:58 2023

@author: orochi
"""

# from pybullet_utils import bullet_client as bc
import pybullet_data
from demos.rl_demo import multiprocess_env
from demos.rl_demo import multiprocess_manipulation_phase
# import rl_env
from demos.rl_demo.multiprocess_state import MultiprocessState
from mojograsp.simcore.goal_holder import  GoalHolder, RandomGoalHolder
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
from stable_baselines3.common.utils import get_device
import wandb
import numpy as np
import time
import os
import multiprocessing
import json

# from stable_baselines3.DQN import MlpPolicy

def make_env(arg_dict=None,rank=0,hand_info=None):
    def _init():
        import pybullet as p1
        env, _, _ = make_pybullet(arg_dict, p1, rank, hand_info)
        return env
    return _init

def make_pybullet(arg_dict, pybullet_instance, rank, hand_info, viz=False):
    # resource paths
    this_path = os.path.abspath(__file__)
    overall_path = os.path.dirname(os.path.dirname(os.path.dirname(this_path)))
    args=arg_dict
    print(args['task'])
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
    elif (args['task'] == 'Rotation_single') | (args['task'] == 'Rotation+Finger'):
        # this will be changed
        # I want to be sure we can rotate the thing in the middle first
        x = [0.0]*1000
        y = [0.0]*1000
        orientations = np.random.uniform(-np.pi/2+0.1, np.pi/2-0.1,1000)
        orientations = orientations + np.sign(orientations)*0.1
        xeval = x
        yeval = y
        eval_orientations = np.random.uniform(-np.pi/2+0.1, np.pi/2-0.1,1000)
        eval_orientations = eval_orientations + np.sign(eval_orientations)*0.1
    elif args['task'] == 'Rotation_region':
        df = pd.read_csv(args['points_path'], index_col=False)
        x = df["x"]/2
        y = df["y"]/2
        df2 = pd.read_csv(overall_path + '/demos/rl_demo/resources/test_points_big.csv', index_col=False)
        xeval = df2['x']/2
        yeval = df2['y']/2
        orientations = np.random.uniform(-np.pi/2+0.1, np.pi/2-0.1,1200)
        orientations = orientations + np.sign(orientations)*0.1
        eval_orientations = np.random.uniform(-np.pi/2+0.1, np.pi/2-0.1,1200)
        eval_orientations = eval_orientations + np.sign(eval_orientations)*0.1
    elif args['task'] == 'slide_and_rotate':
        df = pd.read_csv(args['points_path'], index_col=False)
        x = df["x"]
        y = df["y"]
        df2 = pd.read_csv(overall_path + '/demos/rl_demo/resources/test_points_big.csv', index_col=False)
        xeval = df2['x']
        yeval = df2['y']
        orientations = np.random.uniform(-np.pi/2+0.1, np.pi/2-0.1,1200)
        orientations = orientations + np.sign(orientations)*0.1
        eval_orientations = np.random.uniform(-np.pi/2+0.1, np.pi/2-0.1,1200)
        eval_orientations = eval_orientations + np.sign(eval_orientations)*0.1
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
    
    asterisk_test_points = [[0,0.07],[0.0495,0.0495],[0.07,0.0],[0.0495,-0.0495],[0.0,-0.07],[-0.0495,-0.0495],[-0.07,0.0],[-0.0495,0.0495]]
    names = ['AsteriskSE.pkl','AsteriskS.pkl','AsteriskSW.pkl','AsteriskW.pkl','AsteriskNW.pkl','AsteriskN.pkl','AsteriskNE.pkl','AsteriskE.pkl']
    pose_list = np.array([[i,j] for i,j in zip(x,y)])
    eval_pose_list = [[i,j] for i,j in zip(xeval,yeval)]

    #uncomment for asterisk test
    # pose_list = asterisk_test_points
    # eval_pose_list = asterisk_test_points

    num_eval = len(eval_pose_list)
    eval_pose_list = np.array(eval_pose_list[int(num_eval*rank[0]/rank[1]):int(num_eval*(rank[0]+1)/rank[1])])
    # print(args)
    
    if viz:
        physics_client = pybullet_instance.connect(pybullet_instance.GUI)
    else:
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
    
    this_hand = args['hand_file_list'][rank[0]%len(args['hand_file_list'])]
    hand_type = this_hand.split('/')[0]
    print(hand_type)
    hand_keys = hand_type.split('_')
    info_1 = hand_info[hand_keys[-1]][hand_keys[1]]
    info_2 = hand_info[hand_keys[-1]][hand_keys[2]]
    hand_param_dict = {"link_lengths":[info_1['link_lengths'],info_2['link_lengths']],
                       "starting_angles":[info_1['start_angles'][0],info_1['start_angles'][1],-info_2['start_angles'][0],-info_2['start_angles'][1]],
                       "palm_width":info_1['palm_width'],
                       "hand_name":hand_type}
    # load objects into pybullet
    
    plane_id = pybullet_instance.loadURDF("plane.urdf", flags=pybullet_instance.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
    hand_id = pybullet_instance.loadURDF(args['hand_path'] + '/' + this_hand, useFixedBase=True,
                         basePosition=[0.0, 0.0, 0.05], flags=pybullet_instance.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
    obj_id = pybullet_instance.loadURDF(args['object_path'], basePosition=[0.0, 0.10, .05], flags=pybullet_instance.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
    
    # Create TwoFingerGripper Object and set the initial joint positions
    hand = TwoFingerGripper(hand_id, path=args['hand_path'] + '/' + this_hand,hand_params=hand_param_dict)
    # change visual of gripper
    pybullet_instance.changeVisualShape(hand_id, -1, rgbaColor=[0.3, 0.3, 0.3, 1])
    pybullet_instance.changeVisualShape(hand_id, 0, rgbaColor=[1, 0.5, 0, 1])
    pybullet_instance.changeVisualShape(hand_id, 1, rgbaColor=[0.3, 0.3, 0.3, 1])
    pybullet_instance.changeVisualShape(hand_id, 3, rgbaColor=[1, 0.5, 0, 1])
    pybullet_instance.changeVisualShape(hand_id, 4, rgbaColor=[0.3, 0.3, 0.3, 1])
    obj = ObjectWithVelocity(obj_id, path=args['object_path'],name='obj_2')
    
    # For standard loaded goal poses

    try:
        goal_poses = GoalHolder(pose_list,orientations)
        eval_goal_poses = GoalHolder(eval_pose_list, eval_orientations)
    except:
        if args['task'] == 'unplanned_random':
            goal_poses = RandomGoalHolder([0.02,0.065])
            try:
                eval_goal_poses = GoalHolder(eval_pose_list,goal_names=eval_names)
            except NameError:
                # print('No names')
                eval_goal_poses = GoalHolder(eval_pose_list)
        else:    
            goal_poses = GoalHolder(pose_list)
            try:
                eval_goal_poses = GoalHolder(eval_pose_list,goal_names=eval_names)
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
    
    env = multiprocess_env.MultiprocessSingleShapeEnv(pybullet_instance, hand=hand, obj=obj, hand_type=hand_type, rand_start=args['rstart'])

    # env = rl_env.ExpertEnv(hand=hand, obj=cylinder)
    
    # Create phase
    manipulation = multiprocess_manipulation_phase.MultiprocessManipulation(
        hand, obj, x, y, state, action, reward, env, replay_buffer=replay_buffer, args=arg_dict, hand_type=hand_type)
    
    
    # data recording
    record_data = MultiprocessRecordData(rank,
        data_path=args['save_path'], state=state, action=action, reward=reward, save_all=False, controller=manipulation.controller)
    
    
    gym_env = multiprocess_gym_wrapper.MultiprocessGymWrapper(env, manipulation, record_data, args)
    return gym_env, args, [pose_list,eval_pose_list]

def evaluate(filepath=None,aorb = 'A'):
    num_cpu = multiprocessing.cpu_count() # Number of processes to use
    # Create the vectorized environment
    print('Evaluating on hands A and B')
    print('Hand A: 2v2_50.50_50.50_53')
    print('Hand B: 2v2_65.35_65.35_53')
    actions = np.array([[1, 1, -1, -1], [1, 1, -1, -1], [1, 1, -1, -1], [1, 1, -1, -1], [1, 1, -1, -1], [1, -1, 1, 1], [1, -1, -1, 1], [1, -1, -1, 1], [-1, -1, -1, 1], [-1, -1, -1, 1], [-1, -1, 1, 1], [-1, -1, 1, 1], [-1, -1, 1, 1], [-1, -1, 1, 1], [-1, -1, 1, 1], [0, 0, 0, 0]])

    with open(filepath, 'r') as argfile:
        args = json.load(argfile)
    high_level_folder = os.path.abspath(filepath)
    high_level_folder = os.path.dirname(high_level_folder)
    print(high_level_folder)
    # args['rstart'] = 'N'
    key_file = os.path.abspath(__file__)
    key_file = os.path.dirname(key_file)
    key_file = os.path.join(key_file,'resources','hand_bank','hand_params.json')
    with open(key_file,'r') as hand_file:
        hand_params = json.load(hand_file)
    if args['model'] == 'PPO':
        model_type = PPO
    elif 'DDPG' in args['model']:
        model_type = DDPG
    elif 'TD3' in args['model']:
        model_type = TD3
    print('LOADING A MODEL')

    if aorb =='A':
        args['hand_file_list'] = ["2v2_50.50_50.50_1.1_53/hand/2v2_50.50_50.50_1.1_53.urdf"]
        ht = aorb
    elif aorb =='B':
        args['hand_file_list'] = ["2v2_65.35_65.35_1.1_53/hand/2v2_65.35_65.35_1.1_53.urdf"]
        ht = aorb
    elif aorb.endswith('.urdf'):
        args['hand_file_list'] = [aorb]
        ht = aorb.split('/')[0]
        try:
            folder_to_save = os.path.join(high_level_folder,'Eval_'+ht)
            os.mkdir(folder_to_save)
        except FileExistsError:
            pass
    else:
        print('not going to evaluate, aorb is wrong')
        return
    import pybullet as p2
    eval_env , _, poses= make_pybullet(args,p2, [0,1], hand_params)
    eval_env.evaluate()
    model = model_type("MlpPolicy", eval_env, tensorboard_log=args['tname'], policy_kwargs={'log_std_init':-2.3}).load(args['save_path']+'best_model', env=eval_env)
    for _ in range(1200):
        obs = eval_env.reset()
        done = False
        # print(np.shape(obs))
        thing = 0
        while not done:
            action, _ = model.predict(obs,deterministic=True)
            obs, _, done, _ = eval_env.step(action,hand_type=ht)
            # thing +=1
            # time.sleep(0.5)
        # p2.disconnect()
            # return
            
def replay(argpath, episode_path):
    
    with open(argpath, 'r') as argfile:
        args = json.load(argfile)
    
    key_file = os.path.abspath(__file__)
    key_file = os.path.dirname(key_file)
    key_file = os.path.join(key_file,'resources','hand_bank','hand_params.json')
    with open(key_file,'r') as hand_file:
        hand_params = json.load(hand_file)
    if args['model'] == 'PPO':
        model_type = PPO
    elif 'DDPG' in args['model']:
        model_type = DDPG
    elif 'TD3' in args['model']:
        model_type = TD3
        
        
    with open(episode_path,'rb') as efile:
        data = pkl.load(efile)
    
    actions = [a['action']['actor_output'] for a in data['timestep_list']]
    import pybullet as p2
    eval_env , _, poses= make_pybullet(args,p2, [0,1], hand_params,viz=True)
    eval_env.evaluate()
    start_position = {'goal_position':data['timestep_list'][0]['state']['goal_pose']['goal_position']}
    # model = model_type("MlpPolicy", eval_env, tensorboard_log=args['tname'], policy_kwargs={'log_std_init':-2.3}).load(args['save_path']+'best_model', env=eval_env)
    obs = eval_env.reset(start_position)
    done = False
    # print(np.shape(obs))
    thing = 0
    for act in actions:
        # _, _ = model.predict(obs,deterministic=True)
        obs, _, done, _ = eval_env.step(np.array(act),viz=True)
        print(f'step {thing}')
        thing +=1
        # time.sleep(0.5)
    # p2.disconnect()
        # return
            
def main(filepath = None,learn_type='run'):
    num_cpu = multiprocessing.cpu_count() # Number of processes to use
    # Create the vectorized environment

    if filepath is None:
        filename = 'FTP_full_53'
        filepath = './data/' + filename +'/experiment_config.json'
   
    with open(filepath, 'r') as argfile:
        args = json.load(argfile)
    if num_cpu%len(args['hand_file_list'])!= 0:
        num_cpu = int(int(num_cpu/len(args['hand_file_list']))*len(args['hand_file_list']))
    
    key_file = os.path.abspath(__file__)
    key_file = os.path.dirname(key_file)
    key_file = os.path.join(key_file,'resources','hand_bank','hand_params.json')
    with open(key_file,'r') as hand_file:
        hand_params = json.load(hand_file)
    if args['model'] == 'PPO':
        model_type = PPO
    elif 'DDPG' in args['model']:
        model_type = DDPG
    elif 'TD3' in args['model']:
        model_type = TD3

    vec_env = SubprocVecEnv([make_env(args,[i,num_cpu],hand_info=hand_params) for i in range(num_cpu)])


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

    if learn_type == 'transfer':
        model = model_type("MlpPolicy", vec_env, tensorboard_log=args['tname'], policy_kwargs={'log_std_init':-2.3}).load(args['load_path']+'best_model', env=vec_env)
        print('LOADING A MODEL')
    elif learn_type == 'run':
        model = model_type("MlpPolicy", vec_env,tensorboard_log=args['tname'])

    try:
        print('starting the training using', get_device())
        
        model.learn(total_timesteps=args['epochs']*(args['tsteps']+1), callback=callback)
        filename = os.path.dirname(filepath)
        model.save(filename+'/last_model')

        evaluate(filepath, "A")
        evaluate(filepath, "B")
    except KeyboardInterrupt:
        filename = os.path.dirname(filepath)
        model.save(filename+'/canceled_model')
    # vec_env.env_method('evaluate')
    # for _ in range(1200):
    #     obs =  vec_env.env_method('reset')
    #     done = False
    #     while not done:
    #         action, _ = model.predict(obs, deterministic=True)
    #         obs, _, done, _ = vec_env[0].step(action)

if __name__ == '__main__':
    # evaluate("./data/JA_halfstate_A_rand/experiment_config.json")
    # evaluate("./data/JA_halfstate_A_rand/experiment_config.json","B")
    # evaluate("./data/JA_fullstate_A_rand/experiment_config.json")
    # evaluate("./data/JA_fullstate_A_rand/experiment_config.json","B")
    '''
    filpaths=['./data/JA_fullstate_noise/experiment_config.json','./data/JA_halfstate_noise/experiment_config.json',
              './data/FTP_fullstate_noise/experiment_config.json','./data/FTP_halfstate_noise/experiment_config.json']
    aorbs = ["2v2_50.50_50.50_1.1_53/hand/2v2_50.50_50.50_1.1_53.urdf",
         "2v2_70.30_70.30_1.1_53/hand/2v2_70.30_70.30_1.1_53.urdf",
         "2v2_35.65_35.65_1.1_53/hand/2v2_35.65_35.65_1.1_53.urdf",

         "2v2_65.35_65.35_1.1_63/hand/2v2_65.35_65.35_1.1_63.urdf",
         "2v2_50.50_50.50_1.1_63/hand/2v2_50.50_50.50_1.1_63.urdf",
         "2v2_70.30_70.30_1.1_63/hand/2v2_70.30_70.30_1.1_63.urdf",
         "2v2_35.65_35.65_1.1_63/hand/2v2_35.65_35.65_1.1_63.urdf",
         
         "2v2_65.35_65.35_1.1_73/hand/2v2_65.35_65.35_1.1_73.urdf",
         "2v2_50.50_50.50_1.1_73/hand/2v2_50.50_50.50_1.1_73.urdf",
         "2v2_70.30_70.30_1.1_73/hand/2v2_70.30_70.30_1.1_73.urdf",
         "2v2_35.65_35.65_1.1_73/hand/2v2_35.65_35.65_1.1_73.urdf"]
    JAs = ['Full', 'Half']
    hand_params = ['Hand', 'NoHand']
    Action_space = ['FTP','JA']
    hands = ['BothInterp','PalmExtrap','FingerExtrap']

    things = []
    for k1 in JAs:
        for k2 in hand_params:
            for k3 in Action_space:
                for k4 in hands:
                    temp = '_'.join([k1,k2,k3,k4])
                    things.append(temp)
    precursor = './data/'
    post = '/experiment_config.json'
    for folder_name in things:
        # main(precursor+folder_name+post)
        # evaluate(precursor+folder_name+post,'A')
        # evaluate(precursor+folder_name+post,'B')
        for aorb in aorbs:
            evaluate(precursor+folder_name+post,aorb)
    
    '''
    # main('./data/FTP_halfstate_A_rand_old_finger_poses/experiment_config.json','run')
    # main("./data/JA_newstate_A_rand/experiment_config.json",'run')

    main("./data/JA_finger_reward_region_10_1/experiment_config.json",'run')
    # main("./data/FTP_halfstate_A_rand/experiment_config.json",'run')
    # evaluate("./data/FTP_halfstate_A_rand/experiment_config.json")
    # evaluate("./data/FTP_halfstate_A_rand/experiment_config.json","B")
    # evaluate("./data/FTP_fullstate_A_rand/experiment_config.json")
    # evaluate("./data/FTP_fullstate_A_rand/experiment_config.json","B")
    
    # evaluate("./data/JA_fullstate_A_rand/experiment_config.json")
    # evaluate("./data/JA_fullstate_A_rand/experiment_config.json","B")

    # replay("./data/JA_region_10_1/experiment_config.json", "./data/JA_region_10_1/Eval_A/Episode_16.pkl")


