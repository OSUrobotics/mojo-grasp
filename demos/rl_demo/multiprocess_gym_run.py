#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 10:53:58 2023

@author: orochi
"""

import pybullet_data
from demos.rl_demo import multiprocess_env
from demos.rl_demo import multiprocess_manipulation_phase
# import rl_env
from demos.rl_demo.multiprocess_state import MultiprocessState
from mojograsp.simcore.goal_holder import  GoalHolder, RandomGoalHolder, SingleGoalHolder
from demos.rl_demo import rl_action
from demos.rl_demo import multiprocess_reward
from demos.rl_demo import multiprocess_gym_wrapper
from stable_baselines3.common.vec_env import SubprocVecEnv
import pandas as pd
from demos.rl_demo.multiprocess_record import MultiprocessRecordData
from mojograsp.simobjects.two_finger_gripper import TwoFingerGripper
from mojograsp.simobjects.object_with_velocity import ObjectWithVelocity
from mojograsp.simobjects.multiprocess_object import MultiprocessFixedObject
import pickle as pkl
import json
from stable_baselines3 import TD3, PPO, DDPG, HerReplayBuffer
from stable_baselines3.common.utils import get_device
import numpy as np
import time
import os
import multiprocessing
from scipy.spatial.transform import Rotation as R

def make_env(arg_dict=None,rank=0,hand_info=None):
    def _init():
        import pybullet as p1
        env, _, _ = make_pybullet(arg_dict, p1, rank, hand_info)
        return env
    return _init

def load_set(args):
    print(args['points_path'])
    print(args['test_path'])
    if args['points_path'] =='':
        x = [0.0]
        y = [0.0]
    else:
        df = pd.read_csv(args['points_path'], index_col=False)
        x = df['x']
        y = df['y']
        if 'ang' in df.keys():
            orientations=df['ang']
        else:
            print('NO RANDOM ORIENTATIONS')
            orientations= np.zeros(len(x))
        if 'f1y' in df.keys():
            f1y = df['f1y']
            f2y= df['f2y']
        else:
            f1y = np.random.uniform(-0.01,0.01, len(x))
            f2y = np.random.uniform(-0.01,0.01, len(y))

    if 'test_path' in args.keys():
        df2 = pd.read_csv(args['test_path'],index_col=False)
        xeval = df2['x']
        yeval = df2['y']
        if 'ang' in df2.keys():
            eval_orientations=df2['ang']
        else:
            print('NO RANDOM ORIENTATIONS')
            eval_orientations= np.zeros(len(xeval))
        if 'f1y' in df.keys():
            ef1y = df['f1y']
            ef2y= df['f2y']
        else:
            ef1y = np.random.uniform(-0.01,0.01, len(xeval))
            ef2y = np.random.uniform(-0.01,0.01, len(yeval))
    else:
        xeval = x.copy()
        yeval = y.copy()
        eval_orientations = orientations.copy()
        ef1y = f1y.copy()
        ef2y=f2y.copy()

    if 'contact' in args['task']:
        finger_ys = np.random.uniform( 0.10778391676312778-0.02, 0.10778391676312778+0.02,(len(y),2))
        finger_contacts = np.ones((len(y),4))
        finger_contacts[:,0] = x + 0.026749999999999996
        finger_contacts[:,1] = y + finger_ys[:,0]
        finger_contacts[:,2] = x + -0.026749999999999996
        finger_contacts[:,3] = y + finger_ys[:,1]
        eval_finger_ys = np.random.uniform( 0.10778391676312778-0.02, 0.10778391676312778+0.02,(len(yeval),2))
        eval_finger_contacts = np.ones((len(yeval),4))
        eval_finger_contacts[:,0] = xeval + 0.026749999999999996
        eval_finger_contacts[:,1] = yeval + eval_finger_ys[:,0]
        eval_finger_contacts[:,2] = xeval + -0.026749999999999996
        eval_finger_contacts[:,3] = yeval + eval_finger_ys[:,1]
    else:
        finger_contacts = None
        eval_finger_contacts = None

    pose_list = np.array([[i,j] for i,j in zip(x,y)])
    eval_pose_list = [[i,j] for i,j in zip(xeval,yeval)]
    orientations = [ i for i in orientations]
    eval_orientations = [i for i in eval_orientations]
    f1y = [ i for i in f1y]
    f2y = [i for i in f2y]
    ef1y = [ i for i in ef1y]
    ef2y = [i for i in ef2y]
    
    # print(len(pose_list),len(orientations))
    # print(len(eval_pose_list), len(eval_orientations))
    assert len(pose_list)==len(orientations)
    assert len(eval_pose_list) ==len(eval_orientations)
    # print(f1y)
    return pose_list, eval_pose_list, orientations, eval_orientations, finger_contacts, eval_finger_contacts, [f1y,f2y,ef1y,ef2y]

def load_wall(args):
    df = pd.read_csv(args['points_path'], index_col=False)
    x = df['goal_x']/100
    y = df['goal_y']/100
    wall_x = df['x_wall']
    wall_y = df['y_wall']
    wall_angle = df['ang_wall']
    start_x = df['x_start']
    start_y = df['y_start']
    orientations = [None] * len(x)
    eval_orientations = [None] * len(x)
    f1y = np.random.uniform(-0.01,0.01, len(x))
    f2y = np.random.uniform(-0.01,0.01, len(y))
    ef1y = np.random.uniform(-0.01,0.01, len(x))
    ef2y = np.random.uniform(-0.01,0.01, len(y))
    pose_list = np.array([[i,j] for i,j in zip(x,y)])
    eval_pose_list = [[i,j] for i,j in zip(x,y)]
    orientations = None
    eval_orientations = [None] * len(x)
    f1y = [ i for i in f1y]
    f2y = [i for i in f2y]
    ef1y = [ i for i in ef1y]
    ef2y = [i for i in ef2y]
    finger_contacts = None
    eval_finger_contacts = None 
    return pose_list, eval_pose_list, orientations, eval_orientations, finger_contacts, eval_finger_contacts, [f1y,f2y,ef1y,ef2y]
    
def make_pybullet(arg_dict, pybullet_instance, rank, hand_info, viz=False):
    # resource paths
    this_path = os.path.abspath(__file__)
    overall_path = os.path.dirname(os.path.dirname(os.path.dirname(this_path)))
    args=arg_dict
    # print(args['task'])

    # load the desired test set based on the task
    try:
        pose_list, eval_pose_list, orientations, eval_orientations, finger_contacts, eval_finger_contacts, finger_starts = load_set(args)
    except: 
        pose_list, eval_pose_list, orientations, eval_orientations, finger_contacts, eval_finger_contacts, finger_starts = load_wall(args)
    
    # Break test sets into pieces for multithreading
    num_eval = len(eval_pose_list)
    eval_pose_list = np.array(eval_pose_list[int(num_eval*rank[0]/rank[1]):int(num_eval*(rank[0]+1)/rank[1])])
    eval_orientations = np.array(eval_orientations[int(num_eval*rank[0]/rank[1]):int(num_eval*(rank[0]+1)/rank[1])])
    eval_finger_starts = [finger_starts[2][int(num_eval*rank[0]/rank[1]):int(num_eval*(rank[0]+1)/rank[1])],finger_starts[3][int(num_eval*rank[0]/rank[1]):int(num_eval*(rank[0]+1)/rank[1])]]
    #TODO add the finger contact goal shapes AND the eval finger contact stuff
    # print(type(finger_starts), np.shape(np.array(finger_starts[0:2])))
    # set up goal holders based on task and points given
    if finger_contacts is not None:
        print('we are shuffling the angle and fingertip for the training set WITH A FINGER GOAL')
        eval_finger_contacts = np.array(eval_finger_contacts[int(num_eval*rank[0]/rank[1]):int(num_eval*(rank[0]+1)/rank[1])])
        goal_poses = GoalHolder(pose_list, np.array(finger_starts[0:2]),orientations,finger_contacts, mix_orientation=True, mix_finger=True)
        eval_goal_poses = GoalHolder(eval_pose_list, np.array(eval_finger_starts),eval_orientations,eval_finger_contacts)
    elif orientations is not None:
        print('we are shuffling the angle and fingertip for the training set with no finger goal')
        goal_poses = GoalHolder(pose_list, np.array(finger_starts[0:2]), orientations,mix_orientation=True, mix_finger=True)
        eval_goal_poses = GoalHolder(eval_pose_list, np.array(eval_finger_starts), eval_orientations)
    elif args['task'] == 'unplanned_random':
        goal_poses = RandomGoalHolder([0.02,0.065])
        eval_goal_poses = GoalHolder(eval_pose_list)
    elif args['task'] == 'wall':
        goal_poses = GoalHolder(pose_list, np.array(finger_starts[0:2]))
        eval_goal_poses = GoalHolder(eval_pose_list, np.array(finger_starts[2:4]))
    elif args['task'] == 'wall_single':
        goal_poses = SingleGoalHolder(pose_list)
        eval_goal_poses = SingleGoalHolder(eval_pose_list)
    else:
        goal_poses = GoalHolder(pose_list, finger_starts[0:2])
        eval_goal_poses = GoalHolder(eval_pose_list, finger_starts[2:4])
    
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
            print('older version of object loading, no object domain randomization used')
        else:
            print('normally would get object DR but not this time')
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
        print('STARTING AWAY FROM THE OBJECT')
        hand_param_dict = {"link_lengths":[info_1['link_lengths'],info_2['link_lengths']],
                        "starting_angles":[info_1['near_start_angles'][object_key][0],info_1['near_start_angles'][object_key][1],-info_2['near_start_angles'][object_key][0],-info_2['near_start_angles'][object_key][1]],
                        "palm_width":info_1['palm_width'],
                        "hand_name":hand_type}

    # load objects into pybullet
    plane_id = pybullet_instance.loadURDF("plane.urdf", flags=pybullet_instance.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
    hand_id = pybullet_instance.loadURDF(args['hand_path'] + '/' + this_hand, useFixedBase=True,
                         basePosition=[0.0, 0.0, 0.05], flags=pybullet_instance.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
    print('object path',object_path)
    obj_id = pybullet_instance.loadURDF(object_path, basePosition=[0.0, 0.10, .05], flags=pybullet_instance.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
    print(f'OBJECT ID:{obj_id}')

    # Create TwoFingerGripper Object and set the initial joint positions
    hand = TwoFingerGripper(hand_id, path=args['hand_path'] + '/' + this_hand,hand_params=hand_param_dict)
    
    # change visual of gripper
    pybullet_instance.changeVisualShape(hand_id, -1, rgbaColor=[0.3, 0.3, 0.3, 1])
    pybullet_instance.changeVisualShape(hand_id, 0, rgbaColor=[1, 0.5, 0, 1])
    pybullet_instance.changeVisualShape(hand_id, 1, rgbaColor=[0.3, 0.3, 0.3, 1])
    pybullet_instance.changeVisualShape(hand_id, 3, rgbaColor=[1, 0.5, 0, 1])
    pybullet_instance.changeVisualShape(hand_id, 4, rgbaColor=[0.3, 0.3, 0.3, 1])
    obj = ObjectWithVelocity(obj_id, path=object_path,name='obj_2')


    if 'wall' in args['task']:
        print('LOADING WALL')
        wall_id = pybullet_instance.loadURDF("./resources/object_models/wallthing/vertical_wall.urdf",basePosition=[0.0, 0.10, .05])
        cid = pybullet_instance.createConstraint(wall_id, -1, -1, -1, pybullet_instance.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0.09, 0.02], childFrameOrientation=[ 0, 0, 0.0, 1 ])
        wall = MultiprocessFixedObject(pybullet_instance,wall_id,"./resources/object_models/wallthing/vertical_wall.urdf",'wall')
        state = MultiprocessState(pybullet_instance, objects=[hand, obj, wall, goal_poses], prev_len=args['pv'],eval_goals = eval_goal_poses)
    else:
        # state, action and reward
        state = MultiprocessState(pybullet_instance, objects=[hand, obj, goal_poses], prev_len=args['pv'],eval_goals = eval_goal_poses)
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
    
    if 'wall' in args['task']:
        # maze pybullet enviroment
        env = multiprocess_env.MultiprocessMazeEnv(pybullet_instance, hand, obj, wall, goal_poses, hand_type, args=args)
    else:
        # classic pybullet environment
        env = multiprocess_env.MultiprocessSingleShapeEnv(pybullet_instance, hand=hand, obj=obj, hand_type=hand_type, args=args)
    
    # Create phase
    manipulation = multiprocess_manipulation_phase.MultiprocessManipulation(
        hand, obj, state, action, reward, env, args=arg_dict, hand_type=hand_type)
    
    # data recording
    record_data = MultiprocessRecordData(rank,
        data_path=args['save_path'], state=state, action=action, reward=reward, save_all=False, controller=manipulation.controller)
    
    # gym wrapper around pybullet environment
    gym_env = multiprocess_gym_wrapper.MultiprocessGymWrapper(env, manipulation, record_data, args)
    return gym_env, args, [pose_list,eval_pose_list]



def multiprocess_evaluate_loaded(filepath, aorb):
    # load a trained model and test it on its test set
    print('Evaluating on hands A or B')
    print('Hand A: 2v2_50.50_50.50_53')
    print('Hand B: 2v2_65.35_65.35_53')
    num_cpu =16

    with open(filepath, 'r') as argfile:
        args = json.load(argfile)
    # args['eval-tsteps'] = 20
    high_level_folder = os.path.abspath(filepath)
    high_level_folder = os.path.dirname(high_level_folder)
    print(high_level_folder)
    key_file = os.path.abspath(__file__)
    key_file = os.path.dirname(key_file)
    key_file = os.path.join(key_file,'resources','hand_bank','hand_params.json')
    args['domain_randomization_object_size'] = False
    with open(key_file,'r') as hand_file:
        hand_params = json.load(hand_file)
    if args['model'] == 'PPO':
        model_type = PPO
    elif 'DDPG' in args['model']:
        model_type = DDPG
    elif 'TD3' in args['model']:
        model_type = TD3
    print('LOADING A MODEL')

    # print('HARDCODING THE TEST PATH TO BE THE ROTATION TEST')
    # args['test_path'] ="/home/mothra/mojo-grasp/demos/rl_demo/resources/Solo_rotation_test.csv"

    if not('contact_start' in args.keys()):
        args['contact_start'] = True
        print('we didnt have a contact start so we set it to true')
    if aorb =='A':
        args['hand_file_list'] = ["2v2_50.50_50.50_1.1_53/hand/2v2_50.50_50.50_1.1_53.urdf"]
        ht = aorb
    elif aorb =='B':
        args['hand_file_list'] = ["2v2_65.35_65.35_1.1_53/hand/2v2_65.35_65.35_1.1_53.urdf"]
        ht = aorb
    else:
        print('not going to evaluate, aorb is wrong')
        return
    vec_env = SubprocVecEnv([make_env(args,[i,num_cpu],hand_info=hand_params) for i in range(num_cpu)])
    model = model_type("MlpPolicy", vec_env, tensorboard_log=args['tname'], policy_kwargs={'log_std_init':-2.3}).load(args['save_path']+'best_model', env=vec_env)
    
    if 'Rotation' in args['task']:
        vec_env.env_method('evaluate', aorb)
        for _ in range(int(1200/16)):
            # print('about to reset')
            obs = vec_env.reset()
            # vec_env.env_method()
            done = [False, False]
            while not all(done):
                action, _ = model.predict(obs,deterministic=True)
                vec_env.step_async(action)
                obs, _, done, _ = vec_env.step_wait()
    else:
        df = pd.read_csv('./resources/start_poses.csv', index_col=False)
        x_start = df['x']
        y_start = df['y']
        # input(len(x_start))
        vec_env.env_method('evaluate', aorb)
        for x,y in zip(x_start, y_start):
            tihng = {'goal_position':[x,y]}
            print('THING', tihng)
            vec_env.env_method('set_reset_point', tihng['goal_position'])
            for _ in range(int(1200/16)):
                # print('about to reset')
                obs = vec_env.reset()
                # vec_env.env_method()
                done = [False, False]
                while not all(done):
                    action, _ = model.predict(obs,deterministic=True)
                    vec_env.step_async(action)
                    obs, _, done, _ = vec_env.step_wait()

def multiprocess_evaluate(model, vec_env, rotate=False):
    # vec_env.evaluate()
    # tech_start = time.time()
    if rotate:
        for _ in range(int(1200/16)):
            # print('about to reset')
            obs = vec_env.reset()
            # vec_env.env_method()
            done = [False, False]
            while not all(done):
                action, _ = model.predict(obs,deterministic=True)
                vec_env.step_async(action)
                obs, _, done, _ = vec_env.step_wait()
    else:
        df = pd.read_csv('./resources/start_poses.csv', index_col=False)
        x_start = df['x']
        y_start = df['y']
        vec_env.env_method('evaluate', 'A')
        for x,y in zip(x_start, y_start):
            tihng = {'goal_position':[x,y]}
            print('THING', tihng)
            vec_env.env_method('set_reset_point', tihng['goal_position'])
            for _ in range(int(1200/16)):
                # print('about to reset')
                obs = vec_env.reset()
                # vec_env.env_method()
                done = [False, False]
                while not all(done):
                    action, _ = model.predict(obs,deterministic=True)
                    vec_env.step_async(action)
                    obs, _, done, _ = vec_env.step_wait()
                # print(obs,done)
    # end = time.time()

def evaluate(filepath=None,aorb = 'A'):
    # load a trained model and test it on its test set
    print('Evaluating on hands A or B')
    print('Hand A: 2v2_50.50_50.50_53')
    print('Hand B: 2v2_65.35_65.35_53')
    with open(filepath, 'r') as argfile:
        args = json.load(argfile)
    args['eval-tsteps'] = 30
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
    eval_env , _, poses= make_pybullet(args,p2, [0,1], hand_params, viz=False)
    eval_env.evaluate()
    model = model_type("MlpPolicy", eval_env, tensorboard_log=args['tname'], policy_kwargs={'log_std_init':-2.3}).load(args['save_path']+'best_model', env=eval_env)

    for _ in range(1200):
        obs = eval_env.reset()
        done = False
        # time.sleep(1)
        while not done:
            action, _ = model.predict(obs,deterministic=True)
            obs, _, done, _ = eval_env.step(action,hand_type=ht)
            # time.sleep(0.05)

def mirror_action(filename):
    with open(filename,'rb') as file:
        episode_data = pkl.load(file)

    actions = [[-a['action']['actor_output'][2],-a['action']['actor_output'][3],-a['action']['actor_output'][0],-a['action']['actor_output'][1]] 
               for a in episode_data['timestep_list']]
    return actions   
        
def replay(argpath, episode_path):
    # replays the exact behavior contained in a pkl file without any learning agent running
    # images are saved in videos folder associated with the argfile
    # get parameters from argpath such as action type/size


    with open(argpath, 'r') as argfile:
        args = json.load(argfile)
    if not('contact_start' in args.keys()):
        args['contact_start'] = False
        print('WE DIDNT HAVE A CONTACT START FLAG')
    # load hand parameters (starting angles, link lengths etc)
    args['contact_start'] = False
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
    args['hand_file_list'] = ["2v2_65.35_65.35_1.1_53/hand/2v2_65.35_65.35_1.1_53.urdf"]
    eval_env , _, poses= make_pybullet(args,p2, [1,3], hand_params,viz=True)
    eval_env.evaluate()
    temp = [joint_angles[0]['finger0_segment0_joint'],joint_angles[0]['finger0_segment1_joint'],joint_angles[0]['finger1_segment0_joint'],joint_angles[0]['finger1_segment1_joint']]
    # temp = [-joint_angles[0]['finger1_segment0_joint'],-joint_angles[0]['finger1_segment1_joint'],-joint_angles[0]['finger0_segment0_joint'],-joint_angles[0]['finger0_segment1_joint']]
    obj_temp = data['timestep_list'][0]['state']['goal_pose']['goal_position'].copy()
    # obj_temp[0] = -obj_temp[0]
    # initialize with obeject in desired position. 
    # TODO fix this so that I don't need to comment/uncomment this to get desired behavior
    if ('Rotation' in args['task']) | ('contact' in args['task']):
        start_position = {'goal_position':[-0.04,0.0]}
        # uncomment this line
        #start_position = {'goal_position':obj_temp, 'fingers':temp}

        _ = eval_env.reset(start_position)

    else:
        start_position = {'goal_position':[obj_pose[0][0][0], obj_pose[0][0][1]-0.1]}#, 'fingers':temp}
        _ = eval_env.reset(start_position)
    # print(data['timestep_list'][0]['state']['goal_pose'])
    temp = data['timestep_list'][0]['state']['goal_pose']['goal_position']
    angle = data['timestep_list'][0]['state']['goal_pose']['goal_orientation']
    input('look at it')
    # angle = -data['timestep_list'][0]['state']['goal_pose']['goal_orientation']

    t= R.from_euler('z',angle)
    quat = t.as_quat()
    #obj_temp
    visualShapeId = p2.createVisualShape(shapeType=p2.GEOM_CYLINDER,
                                        rgbaColor=[1, 0, 0, 1],
                                        radius=0.004,
                                        length=0.02,
                                        specularColor=[0.4, .4, 0],
                                        visualFramePosition=[[obj_temp[0],obj_temp[1]+0.1,0.1]],
                                        visualFrameOrientation=[ 0.7071068, 0, 0, 0.7071068 ])
    collisionShapeId = p2.createCollisionShape(shapeType=p2.GEOM_CYLINDER,
                                            radius=0.002,
                                            height=0.002,)

    tting = p2.createMultiBody(baseMass=0,
                    baseInertialFramePosition=[0,0,0],
                    baseCollisionShapeIndex=collisionShapeId,
                    baseVisualShapeIndex=visualShapeId,
                    basePosition=[obj_temp[0]-0.0025,obj_temp[1]+0.1-0.0025,0.11],
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
    print('starting position', f1_poses[0],f2_poses[0], joint_angles[0])
    # input('start')
    joints = []
    for i,act in enumerate(actions):
        # print('action vs mirrored:', actions[i],act)
        print('joints in pkl file',joint_angles[i])
        print(f'step {i}')
        print(act)
        eval_env.step(np.array(act),viz=True)
        step_num +=1
        print('reward from pickle', data['timestep_list'][i]['reward'])
        # input('next step?')
        # time.sleep(0.5)
        # print(f'finger poses in pkl file, {f1_poses[i]}, {f2_poses[i]}')
        # print(data['timestep_list'][i]['action'])
        # input('next step?')
        # joints.append()

def main(filepath = None,learn_type='run'):
    num_cpu = 16 #multiprocessing.cpu_count() # Number of processes to use
    # Create the vectorized environment
    print('cuda y/n?', get_device())
    if filepath is None:
        filename = 'FTP_full_53'
        filepath = './data/' + filename +'/experiment_config.json'
   
    with open(filepath, 'r') as argfile:
        args = json.load(argfile)

    # TEMPORARY, REMOVE AT START OF JUNE 2024
    if not('contact_start' in args.keys()):
        args['contact_start'] = True
        print('WE DIDNT HAVE A CONTACT START FLAG, setting contact start to true')

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

    train_timesteps = int(args['evaluate']*(args['tsteps']+1)/num_cpu)
    callback = multiprocess_gym_wrapper.MultiEvaluateCallback(vec_env,n_eval_episodes=int(1200), eval_freq=train_timesteps, best_model_save_path=args['save_path'])

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

        # multiprocess_evaluate(model,vec_env)
    except KeyboardInterrupt:
        filename = os.path.dirname(filepath)
        model.save(filename+'/canceled_model')

if __name__ == '__main__':
    # main('./data/HPC_slide_all_randomizations/FTP_S1/experiment_config.json')
    # main('./data/HPC_slide_time_tests/25_no_contact/experiment_config.json')
    # main('./data/Full_long_test/experiment_config.json')
    # main('./data/Solo_rotation_no_pose_random/experiment_config.json')
    # main("./data/region_rotation_JA_finger/experiment_config.json",'run')
    # main("./data/JA_full_task_20_1/experiment_config.json",'run')
    # evaluate("./data/FTP_halfstate_A_rand/experiment_config.json","B")
    # evaluate("./data/FTP_fullstate_A_rand/experiment_config.json")
    # evaluate("./data/FTP_fullstate_A_rand/experiment_config.json","B")
    # evaluate("./data/Domain_randomization_test/experiment_config.json")
    # evaluate("./data/JA_halfstate_A_rand/experiment_config.json", "B")
    # evaluate("./data/JA_fullstate_A_rand/experiment_config.json","B")
    # replay("./data/Mothra_Rotation/JA_S2_no_contact/experiment_config.json","./data/Mothra_Rotation/JA_S2_no_contact/Eval_A/Episode_124.pkl")
    # main("./data/Full_task_hyperparameter_search/JA_1-3/experiment_config.json",'run')
    # replay("./data/Mothra_Slide/JA_S2/experiment_config.json","./data/Mothra_Slide/JA_S2/Eval_A/Episode_2.pkl")
    # multiprocess_evaluate_loaded("./data/Mothra_Rotation/JA_S2_no_contact/experiment_config.json","A")
    # multiprocess_evaluate_loaded("./data/Mothra_Rotation/FTP_S1/experiment_config.json","A")
    # multiprocess_evaluate_loaded("./data/Rotation_no_finger/experiment_config.json","A")
    # multiprocess_evaluate_loaded("./data/Rotation_no_finger/experiment_config.json","B")
    # multiprocess_evaluate_loaded("./data/HPC_slide_time_tests/25_no_contact/experiment_config.json","A")
    # multiprocess_evaluate_loaded("./data/full_15_contact/experiment_config.json","A")
    # multiprocess_evaluate_loaded("./data/Mothra_Rotation/JA_S2_no_contact/experiment_config.json","A")
    # multiprocess_evaluate_loaded("./data/Mothra_Slide/JA_S3/experiment_config.json","A")
    # multiprocess_evaluate_loaded("./data/Mothra_Slide/JA_S3/experiment_config.json","B")

    # multiprocess_evaluate_loaded("./data/Full_length_test/full_finger_long_train/experiment_config.json","A")
    # multiprocess_evaluate_loaded("./data/Full_length_test/full_finger_long_train/experiment_config.json","B")
  
    # multiprocess_evaluate_loaded("./data/WhateverI want/experiment_config.json","A")
    # multiprocess_evaluate_loaded("./data/Rotation_length_tests/Long_finger/experiment_config.json","B")
    # multiprocess_evaluate_loaded("./data/Mothra_Rotation/FTP_S3/experiment_config.json","A")
    # multiprocess_evaluate_loaded("./data/Mothra_Rotation/FTP_S3/experiment_config.json","B")
    # multiprocess_evaluate_loaded("./data/Mothra_Rotation/FTP_S2/experiment_config.json","A")
    # multiprocess_evaluate_loaded("./data/Mothra_Rotation/FTP_S2/experiment_config.json","B")

    # multiprocess_evaluate_loaded("./data/Misc_Slide/JA_S1/experiment_config.json","A")
    # multiprocess_evaluate_loaded("./data/Misc_Slide/JA_S1/experiment_config.json","B")
    # multiprocess_evaluate_loaded("./data/Misc_Slide/JA_S2/experiment_config.json","A")
    # multiprocess_evaluate_loaded("./data/Misc_Slide/JA_S2/experiment_config.json","B")

    # multiprocess_evaluate_loaded("./data/Misc_Slide/JA_S3/experiment_config.json","A")
    # multiprocess_evaluate_loaded("./data/Misc_Slide/JA_S3/experiment_config.json","B")
    # multiprocess_evaluate_loaded("./data/Misc_Slide/FTP_S3/experiment_config.json","A")
    # multiprocess_evaluate_loaded("./data/Misc_Slide/FTP_S3/experiment_config.json","B")

    # multiprocess_evaluate_loaded("./data/Misc_Slide/FTP_S1/experiment_config.json","A")
    # multiprocess_evaluate_loaded("./data/Misc_Slide/FTP_S1/experiment_config.json","B")
    # multiprocess_evaluate_loaded("./data/Misc_Slide/FTP_S2/experiment_config.json","A")
    # multiprocess_evaluate_loaded("./data/Misc_Slide/FTP_S2/experiment_config.json","B")

    # multiprocess_evaluate_loaded("./data/Mothra_Slide/FTP_S3/experiment_config.json","A")
    # multiprocess_evaluate_loaded("./data/Mothra_Slide/FTP_S3/experiment_config.json","B")
    # multiprocess_evaluate_loaded("./data/Mothra_Slide/JA_S1/experiment_config.json","A")
    # multiprocess_evaluate_loaded("./data/Mothra_Slide/JA_S1/experiment_config.json","B")

    # multiprocess_evaluate_loaded("./data/Mothra_Slide/JA_S2/experiment_config.json","A")
    # multiprocess_evaluate_loaded("./data/Mothra_Slide/JA_S2/experiment_config.json","B")

    multiprocess_evaluate_loaded("./data/Jeremiah_Rotation/FTP_S1/experiment_config.json","A")
    multiprocess_evaluate_loaded("./data/Jeremiah_Rotation/FTP_S1/experiment_config.json","B")
    multiprocess_evaluate_loaded("./data/Jeremiah_Rotation/FTP_S2/experiment_config.json","A")
    multiprocess_evaluate_loaded("./data/Jeremiah_Rotation/FTP_S2/experiment_config.json","B")
    multiprocess_evaluate_loaded("./data/Jeremiah_Rotation/FTP_S3/experiment_config.json","A")
    multiprocess_evaluate_loaded("./data/Jeremiah_Rotation/FTP_S3/experiment_config.json","B")
    # multiprocess_evaluate_loaded("./data/HPC_slide_all_randomizations/JA_S3/experiment_config.json","A")
    # multiprocess_evaluate_loaded("./data/HPC_slide_all_randomizations/FTP_S1/experiment_config.json","A")