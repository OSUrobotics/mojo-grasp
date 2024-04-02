#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 10:53:58 2023

@author: orochi
"""

# from pybullet_utils import bullet_client as bc
import pybullet_data
from demos.rl_demo import multiprocess_direction_env
from demos.rl_demo import multiprocess_direction_phase
# import rl_env
from demos.rl_demo.multiprocess_state import MultiprocessState
from mojograsp.simcore.goal_holder import  GoalHolder, RandomGoalHolder, SimpleGoalHolder
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
# from stable_baselines3.DQN import MlpPolicy

def make_env(arg_dict=None,rank=0,hand_info=None, goal_dir=None):
    def _init():
        import pybullet as p1
        env, _ = make_pybullet(arg_dict, p1, rank, hand_info, goal_dir)
        return env
    return _init

def make_HRL_env(arg_dict=None,rank=0,hand_info=None,sub_policies=None):
    def _init():
        import pybullet as p1
        env, _ = make_HRL_pybullet(arg_dict, p1, rank, hand_info,sub_policies)
        return env
    return _init

def make_pybullet(arg_dict, pybullet_instance, rank, hand_info, goal_dir, viz=False):
    # resource paths
    this_path = os.path.abspath(__file__)
    overall_path = os.path.dirname(os.path.dirname(os.path.dirname(this_path)))
    args=arg_dict
    theta = np.random.uniform(0, 2*np.pi,1000)
    r = (1-(np.random.uniform(0, 0.95,1000))**2) * 50/1000
    x = r * np.sin(theta)
    y = r * np.cos(theta)
    obj_pos = np.array([[i,j] for i,j in zip(x,y)])
    fingers = np.random.uniform(0.01,0.01,(1000,2))

    goals = SimpleGoalHolder(goal_dir)

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
    this_hand = args['hand_file_list'][rank[0]%len(args['hand_file_list'])]
    hand_type = this_hand.split('/')[0]
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
    print(f'OBJECT ID:{obj_id}')
    # Create TwoFingerGripper Object and set the initial joint positions
    hand = TwoFingerGripper(hand_id, path=args['hand_path'] + '/' + this_hand,hand_params=hand_param_dict)
    
    # change visual of gripper
    pybullet_instance.changeVisualShape(hand_id, -1, rgbaColor=[0.3, 0.3, 0.3, 1])
    pybullet_instance.changeVisualShape(hand_id, 0, rgbaColor=[1, 0.5, 0, 1])
    pybullet_instance.changeVisualShape(hand_id, 1, rgbaColor=[0.3, 0.3, 0.3, 1])
    pybullet_instance.changeVisualShape(hand_id, 3, rgbaColor=[1, 0.5, 0, 1])
    pybullet_instance.changeVisualShape(hand_id, 4, rgbaColor=[0.3, 0.3, 0.3, 1])
    obj = ObjectWithVelocity(obj_id, path=args['object_path'],name='obj_2')

    # state, action and reward
    state = MultiprocessState(pybullet_instance, objects=[hand, obj, goals], prev_len=args['pv'])
    if args['freq'] ==240:
        action = rl_action.ExpertAction()
    else:
        action = rl_action.InterpAction(args['freq'])
    reward = multiprocess_direction_reward.MultiprocessDirectionReward(pybullet_instance)

    #change initial physics parameters
    pybullet_instance.changeDynamics(plane_id,-1,lateralFriction=0.05, spinningFriction=0.05, rollingFriction=0.05)
    pybullet_instance.changeDynamics(obj.id, -1, mass=.03, restitution=.95, lateralFriction=1)
    
    # set up dictionary for manipulation phase
    arg_dict = args.copy()
    if args['action'] == 'Joint Velocity':
        arg_dict['ik_flag'] = False
    else:
        arg_dict['ik_flag'] = True

    # pybullet environment
    env = multiprocess_direction_env.MultiprocessDirectionSingleShapeEnv(pybullet_instance, hand=hand, obj=obj, args=args, obj_starts=obj_pos, finger_ys=fingers)

    # Create phase
    manipulation = multiprocess_direction_phase.MultiprocessManipulation(
        hand, obj, state, action, reward, env, args=arg_dict, hand_type=hand_type)
    
    # data recording
    record_data = MultiprocessRecordData(rank,
        data_path=args['save_path'], state=state, action=action, reward=reward, save_all=False, controller=manipulation.controller)
    
    # gym wrapper around pybullet environment
    gym_env = multiprocess_hierarchical_wrapper.MultiprocessGymWrapper(env, manipulation, record_data, args)
    return gym_env, args

def make_HRL_pybullet(arg_dict, pybullet_instance, rank, hand_info, sub_policies, viz=False):
    # resource paths
    this_path = os.path.abspath(__file__)
    overall_path = os.path.dirname(os.path.dirname(os.path.dirname(this_path)))
    args=arg_dict
    theta = np.random.uniform(0, 2*np.pi,1000)
    r = (1-(np.random.uniform(0, 0.95,1000))**2) * 50/1000
    x = r * np.sin(theta)
    y = r * np.cos(theta)
    obj_pos = np.array([[i,j] for i,j in zip(x,y)])
    theta = np.random.uniform(0, 2*np.pi,1000)
    r = (1-(np.random.uniform(0, 0.95,1000))**2) * 50/1000
    x = r * np.sin(theta)
    y = r * np.cos(theta)
    obj_goal = np.array([[i,j] for i,j in zip(x,y)])
    fingers = np.random.uniform(0.01,0.01,(1000,2))

    goals = GoalHolder(obj_goal, fingers)

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
    this_hand = args['hand_file_list'][rank[0]%len(args['hand_file_list'])]
    hand_type = this_hand.split('/')[0]
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
    print(f'OBJECT ID:{obj_id}')
    # Create TwoFingerGripper Object and set the initial joint positions
    hand = TwoFingerGripper(hand_id, path=args['hand_path'] + '/' + this_hand,hand_params=hand_param_dict)
    
    # change visual of gripper
    pybullet_instance.changeVisualShape(hand_id, -1, rgbaColor=[0.3, 0.3, 0.3, 1])
    pybullet_instance.changeVisualShape(hand_id, 0, rgbaColor=[1, 0.5, 0, 1])
    pybullet_instance.changeVisualShape(hand_id, 1, rgbaColor=[0.3, 0.3, 0.3, 1])
    pybullet_instance.changeVisualShape(hand_id, 3, rgbaColor=[1, 0.5, 0, 1])
    pybullet_instance.changeVisualShape(hand_id, 4, rgbaColor=[0.3, 0.3, 0.3, 1])
    obj = ObjectWithVelocity(obj_id, path=args['object_path'],name='obj_2')

    # state, action and reward
    state = MultiprocessState(pybullet_instance, objects=[hand, obj, goals], prev_len=args['pv'])
    if args['freq'] ==240:
        action = rl_action.ExpertAction()
    else:
        action = rl_action.InterpAction(args['freq'])
    reward = multiprocess_direction_reward.MultiprocessDirectionReward(pybullet_instance)

    #change initial physics parameters
    pybullet_instance.changeDynamics(plane_id,-1,lateralFriction=0.05, spinningFriction=0.05, rollingFriction=0.05)
    pybullet_instance.changeDynamics(obj.id, -1, mass=.03, restitution=.95, lateralFriction=1)
    
    # set up dictionary for manipulation phase
    arg_dict = args.copy()
    if args['action'] == 'Joint Velocity':
        arg_dict['ik_flag'] = False
    else:
        arg_dict['ik_flag'] = True

    # pybullet environment
    env = multiprocess_direction_env.MultiprocessDirectionSingleShapeEnv(pybullet_instance, hand=hand, obj=obj, args=args, obj_starts=obj_pos, finger_ys=fingers)

    # Create phase
    manipulation = multiprocess_direction_phase.MultiprocessManipulation(
        hand, obj, state, action, reward, env, args=arg_dict, hand_type=hand_type)
    
    # data recording
    record_data = MultiprocessRecordData(rank,
        data_path=args['save_path'], state=state, action=action, reward=reward, save_all=False, controller=manipulation.controller)
    
    # gym wrapper around pybullet environment
    gym_env = multiprocess_hierarchical_wrapper.SimpleHRLWrapper(env, manipulation, record_data, sub_policies, args)
    return gym_env, args


def evaluate(filepath=None,aorb = 'A'):
    # load a trained model and test it on its test set
    print('Evaluating on hands A and B')
    print('Hand A: 2v2_50.50_50.50_53')
    print('Hand B: 2v2_65.35_65.35_53')

    with open(filepath, 'r') as argfile:
        args = json.load(argfile)
    high_level_folder = os.path.abspath(filepath)
    high_level_folder = os.path.dirname(high_level_folder)
    print(high_level_folder)
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
        while not done:
            action, _ = model.predict(obs,deterministic=True)
            obs, _, done, _ = eval_env.step(action,hand_type=ht)

        
def replay(argpath, episode_path):
    # replays the exact behavior contained in a pkl file without any learning agent running
    # images are saved in videos folder associated with the argfile

    # get parameters from argpath such as action type/size
    with open(argpath, 'r') as argfile:
        args = json.load(argfile)
    
    # load hand parameters (starting angles, link lengths etc)
    key_file = os.path.abspath(__file__)
    key_file = os.path.dirname(key_file)
    key_file = os.path.join(key_file,'resources','hand_bank','hand_params.json')
    with open(key_file,'r') as hand_file:
        hand_params = json.load(hand_file)
    
    # load episode data
    with open(episode_path,'rb') as efile:
        data = pkl.load(efile)
    mirrored = mirror_action(episode_path)
    actions = [a['action']['actor_output'] for a in data['timestep_list']]
    obj_pose = [s['state']['obj_2']['pose'] for s in data['timestep_list']]
    f1_poses = [s['state']['f1_pos'] for s in data['timestep_list']]
    f2_poses = [s['state']['f2_pos'] for s in data['timestep_list']]
    joint_angles = [s['state']['two_finger_gripper']['joint_angles'] for s in data['timestep_list']]
    import pybullet as p2
    eval_env , _, poses= make_pybullet(args,p2, [0,1], hand_params,viz=True)
    eval_env.evaluate()
    temp = [joint_angles[0]['finger0_segment0_joint'],joint_angles[0]['finger0_segment1_joint'],joint_angles[0]['finger1_segment0_joint'],joint_angles[0]['finger1_segment1_joint']]
    # initialize with obeject in desired position. 
    # TODO fix this so that I don't need to comment/uncomment this to get desired behavior
    if ('Rotation' in args['task']) | ('contact' in args['task']):
        start_position = {'goal_position':data['timestep_list'][0]['state']['goal_pose']['goal_position'], 'fingers':temp}

        _ = eval_env.reset(start_position)

    else:
        _ = eval_env.reset()
    print(data['timestep_list'][0]['state']['goal_pose'])
    temp = data['timestep_list'][0]['state']['goal_pose']['goal_position']
    angle = data['timestep_list'][0]['state']['goal_pose']['goal_orientation']

    
    t= R.from_euler('z',angle)
    quat = t.as_quat()

    visualShapeId = p2.createVisualShape(shapeType=p2.GEOM_CYLINDER,
                                        rgbaColor=[1, 0, 0, 1],
                                        radius=0.004,
                                        length=0.02,
                                        specularColor=[0.4, .4, 0],
                                        visualFramePosition=[[temp[0],temp[1]+0.1,0.1]],
                                        visualFrameOrientation=[ 0.7071068, 0, 0, 0.7071068 ])
    collisionShapeId = p2.createCollisionShape(shapeType=p2.GEOM_CYLINDER,
                                            radius=0.002,
                                            height=0.002,)

    tting = p2.createMultiBody(baseMass=0,
                    baseInertialFramePosition=[0,0,0],
                    baseCollisionShapeIndex=collisionShapeId,
                    baseVisualShapeIndex=visualShapeId,
                    basePosition=[temp[0]-0.0025,temp[1]+0.1-0.0025,0.11],
                    baseOrientation =quat,
                    useMaximalCoordinates=True)
    
    temp_pos = obj_pose[0][0].copy()
    temp_pos[2] += 0.06
    curr_id=p2.loadURDF('./resources/object_models/2v2_mod/2v2_mod_cylinder_small_alt.urdf', flags=p2.URDF_ENABLE_CACHED_GRAPHICS_SHAPES,
                globalScaling=0.2, basePosition=temp_pos, baseOrientation=[ 0.7071068, 0, 0, 0.7071068 ])
    p2.changeVisualShape(curr_id, -1,rgbaColor=[1, 0.5, 0, 1])
    cid = p2.createConstraint(2, -1, curr_id, -1, p2.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0,0.06,0], childFrameOrientation=[ 0.7071068, 0, 0, 0.7071068 ])
    p2.setCollisionFilterPair(curr_id,tting,-1,-1,0)
    

    if 'contact' in args['task']:
        temp = data['timestep_list'][0]['state']['goal_pose']['goal_finger'][0:2]
        visualShapeId = p2.createVisualShape(shapeType=p2.GEOM_SPHERE,
                                            rgbaColor=[0, 1, 0, 1],
                                            radius=0.005,
                                            specularColor=[0.4, .4, 0],
                                            visualFramePosition=[[temp[0],temp[1],0.1]])
        collisionShapeId = p2.createCollisionShape(shapeType=p2.GEOM_SPHERE,
                                                radius=0.001)

        p2.createMultiBody(baseMass=0,
                        baseInertialFramePosition=[0,0,0],
                        baseCollisionShapeIndex=collisionShapeId,
                        baseVisualShapeIndex=visualShapeId,
                        basePosition=[temp[0]-0.0025,temp[1]-0.0025,0.11],
                        useMaximalCoordinates=True)
        
        
        
        temp = data['timestep_list'][0]['state']['goal_pose']['goal_finger'][2:4]
        visualShapeId = p2.createVisualShape(shapeType=p2.GEOM_SPHERE,
                                            rgbaColor=[0, 0, 1, 1],
                                            radius=0.005,
                                            specularColor=[0.4, .4, 0],
                                            visualFramePosition=[[temp[0],temp[1],0.1]])
        collisionShapeId = p2.createCollisionShape(shapeType=p2.GEOM_SPHERE,
                                                radius=0.001)

        p2.createMultiBody(baseMass=0,
                        baseInertialFramePosition=[0,0,0],
                        baseCollisionShapeIndex=collisionShapeId,
                        baseVisualShapeIndex=visualShapeId,
                        basePosition=[temp[0]-0.0025,temp[1]-0.0025,0.11],
                        useMaximalCoordinates=True)

    p2.configureDebugVisualizer(p2.COV_ENABLE_RENDERING,1)
    step_num = 0
    input('start')
    for i,act in enumerate(mirrored):
        print('action vs mirrored:', actions[i],act)
        print('joints in pkl file',joint_angles[i+1])
        eval_env.step(np.array(act),viz=True)
        step_num +=1
        time.sleep(0.5)
        # print(f'finger poses in pkl file, {f1_poses[i+1]}, {f2_poses[i]}')
        # print(data['timestep_list'][i]['action'])
        # input('next step?')

def main(filepath = None, train_type='pre'):
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

    names= ["N","NE","E","SE","S","SW","W","NW"]
    red_names = ['E','S','W']
    directions  = [[np.sin(np.pi*i/4), np.cos(np.pi*i/4)] for i in range(8)]
    red_dir = [[np.sin(np.pi*i/2), np.cos(np.pi*i/2)] for i in range(1,4)]
    if train_type =='pre':
        for name, direction in zip(red_names, red_dir):
            vec_env = SubprocVecEnv([make_env(args,[i,num_cpu],hand_info=hand_params,goal_dir=direction) for i in range(num_cpu)])

            # 
            model = model_type("MlpPolicy", vec_env,tensorboard_log=args['tname'])

            try:
                print('starting the training using', get_device())
                model.learn(total_timesteps=100000*(args['tsteps']+1))
                filename = os.path.dirname(filepath)
                model.save(filename+'/last_model_'+name)

            except KeyboardInterrupt:
                filename = os.path.dirname(filepath)
                model.save(filename+'/canceled_model_'+name)
    else:
        filename = os.path.dirname(filepath)
        policy_folder = os.listdir(filename)
        names = [p for p in policy_folder if 'last_model' in p]
        subpolicies = []
        direction = [0,0]
        import pybullet as p1
        for name in names:
            model = model_type("MlpPolicy", None, _init_setup_model=False).load(filename+'/'+name)
            subpolicies.append(model)
        env, _ = make_HRL_pybullet(args,pybullet_instance=p1, rank=[0,1],hand_info=hand_params,sub_policies=subpolicies)
        # callback = multiprocess_gym_wrapper.MultiEvaluateCallback(vec_env,n_eval_episodes=int(1200), eval_freq=train_timesteps, best_model_save_path=args['save_path'])
        model = model_type('MlpPolicy',env,  tensorboard_log=args['tname'], policy_kwargs={'log_std_init':-2.3})
        try:
            print('starting the training using', get_device())
            model.learn(total_timesteps=500000*(args['tsteps']+1))
            filename = os.path.dirname(filepath)
            model.save(filename+'/last_model_full')

        except KeyboardInterrupt:
            filename = os.path.dirname(filepath)
            model.save(filename+'/canceled_model_full')
if __name__ == '__main__':
    main('./data/HRL_test_1/experiment_config.json', 'getfucked')
