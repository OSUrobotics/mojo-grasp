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
from mojograsp.simcore.start_holder import StartHolder,RandomStartHolder
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
from demos.rl_demo.pkl_merger import merge_from_folder
from scipy.spatial.transform import Rotation as R
from stable_baselines3.common.noise import NormalActionNoise
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy
import math
import demos.rl_demo.point_generator as pg



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
    
def make_pybullet(arg_dict, pybullet_instance, rank, hand_info, frictionList = None, contactList = None, viz=False):
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
        goal_poses = GoalHolder(pose_list,orientations,finger_contacts, mix_orientation=True, mix_finger=True)
        eval_goal_poses = GoalHolder(eval_pose_list,eval_orientations,eval_finger_contacts)
    elif orientations is not None:
        print('we are shuffling the angle and fingertip for the training set with no finger goal')
        goal_poses = GoalHolder(pose_list, orientations,mix_orientation=True, mix_finger=True)
        eval_goal_poses = GoalHolder(eval_pose_list, eval_orientations)
    elif args['task'] == 'unplanned_random':
        goal_poses = RandomGoalHolder([0.02,0.065])
        eval_goal_poses = GoalHolder(eval_pose_list)
    elif args['task'] == 'wall':
        goal_poses = GoalHolder(pose_list)
        eval_goal_poses = GoalHolder(eval_pose_list)
    elif args['task'] == 'wall_single':
        goal_poses = SingleGoalHolder(pose_list)
        eval_goal_poses = SingleGoalHolder(eval_pose_list)
    else:
        goal_poses = GoalHolder(pose_list)
        eval_goal_poses = GoalHolder(eval_pose_list)
    # df = pd.read_csv('INSERTCSVHERE.csv', index_col=False)
    # x=df['x']
    # y=df['y']
    # object_orientation = df['theta']
    # object_pos = np.array([[i,j] for i,j in zip(x,y)])
    # y1= df['f1y']
    # y2= df['f2y']
    # finger_starts = np.array([[i,j] for i,j in zip(y1,y2)])
    try:
        start_orientation_ranges = [args['starting_orientation_low'],args['starting_orientation_high']]
    except:
        start_orientation_ranges = [0,0]
    random_start_dict = {'orientation':start_orientation_ranges}
    if 'starting_x_low' in args.keys():
        start_x_ranges = [args['starting_x_low'], args['starting_x_high']]
        start_y_ranges = [args['starting_y_low'], args['starting_y_high']]
        random_start_dict['x'] = start_x_ranges
        random_start_dict['y'] = start_y_ranges
    elif 'starting_r_low' in args.keys():
        start_r_ranges = [args['starting_r_low'], args['starting_r_high']]
        start_theta_ranges = [args['starting_theta_low'], args['starting_theta_high']]
        random_start_dict['r'] = start_r_ranges
        random_start_dict['theta'] = start_theta_ranges
    else:
        start_r_ranges = [0, 0]
        start_theta_ranges = [0,0]
        random_start_dict['r'] = start_r_ranges
        random_start_dict['theta'] = start_theta_ranges

    if 'starting_finger_y_low' in args.keys():
        random_start_dict['fingery'] = [args['starting_finger_y_low'], args['starting_finger_y_high']]
    start_holder = RandomStartHolder(random_start_dict)
    eval_start_holder = RandomStartHolder(random_start_dict)
    # setup pybullet client to either run with or without rendering
    if viz:
        physics_client = pybullet_instance.connect(pybullet_instance.GUI)
    else:
        physics_client = pybullet_instance.connect(pybullet_instance.DIRECT)

    # set initial gravity and general features
    pybullet_instance.resetSimulation()
    pybullet_instance.setAdditionalSearchPath(pybullet_data.getDataPath())
    pybullet_instance.setGravity(0, 0, -10)
    pybullet_instance.setPhysicsEngineParameter(contactBreakingThreshold=.001, contactERP=0.8, enableConeFriction=1, globalCFM=0.01, numSubSteps=1)
    pybullet_instance.setRealTimeSimulation(0)
    pybullet_instance.resetDebugVisualizerCamera(cameraDistance=.02, cameraYaw=0, cameraPitch=-89.9999,
                                 cameraTargetPosition=[0, 0.1, 0.5])
    pybullet_instance.configureDebugVisualizer(pybullet_instance.COV_ENABLE_MOUSE_PICKING,0)
    pybullet_instance.configureDebugVisualizer(pybullet_instance.COV_ENABLE_WIREFRAME,0)
    
    # load hand/hands 
    if rank[1] < len(args['hand_file_list']):
        raise IndexError('TOO MANY HANDS FOR NUMBER OF PROVIDED CORES')
    elif rank[1] % len(args['hand_file_list']) != 0:
        print('WARNING: number of hands does not evenly divide into number of pybullet instances. Hands will have uneven number of samples')
    try:
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
        elif args['random_shapes']:
            object_path = args['object_path'][rank[0]%len(args['object_path'])]
            if 'square' in object_path:
                if 'small' in object_path:
                    object_key = 'sub10'
                elif 'large' in object_path:
                    object_key = 'add10'
                else:
                    object_key = 'small'
            if 'circle' in object_path:
                if 'small' in object_path:
                    object_key = 'sub10'
                elif 'large' in object_path:
                    object_key = 'add10'
                else:
                    object_key = 'small'
            if 'triangle' in object_path:
                if 'small' in object_path:
                    object_key = 'sub10'
                elif 'large' in object_path:
                    object_key = 'add10'
                else:
                    object_key = 'small'
            # if 'ellipse' in object_path:
            #     if 'small' in object_path:
            #         object_key = 'small_ellipse'
            #     elif 'large' in object_path:
            #         object_key = 'large_ellipse'
            #     else:
            #         object_key = 'medium_ellipse'
            # if 'teardrop' in object_path:
            #     if 'small' in object_path:
            #         object_key = 'small_teardrop'
            #     elif 'large' in object_path:
            #         object_key = 'large_teardrop'
            #     else:
            #         object_key = 'medium_teardrop'
            # if 'concave' in object_path:
            #     if 'small' in object_path:
            #         object_key = 'small_concave'
            #     elif 'large' in object_path:
            #         object_key = 'large_concave'
            #     else:
            #         object_key = 'medium_concave'
        else:
            if type(args['object_path']) == str:
                object_path = args['object_path']
                object_key = "small"
                # print('older version of object loading, no object domain randomization used')
            else:
                #print('WE ARE HERE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                object_path = args['object_path'][rank[0]%len(args['object_path'])]
                object_key = 'small'   

    except KeyError:
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
                #print('WE ARE HERE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
                object_path = args['object_path'][2]
                object_key = 'add10'   

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
        print('STARTING AWAY FROM THE OBJECT', object_key)
        hand_param_dict = {"link_lengths":[info_1['link_lengths'],info_2['link_lengths']],
                        "starting_angles":[info_1['near_start_angles'][object_key][0],info_1['near_start_angles'][object_key][1],-info_2['near_start_angles'][object_key][0],-info_2['near_start_angles'][object_key][1]],
                        "palm_width":info_1['palm_width'],
                        "hand_name":hand_type}
        
    # load objects into pybullet
    plane_id = pybullet_instance.loadURDF("plane.urdf", flags=pybullet_instance.URDF_ENABLE_CACHED_GRAPHICS_SHAPES, basePosition=[0.50,0.50,0])
    # other_id = pybullet_instance.loadURDF('./resources/object_models/wallthing/vertical_wall.urdf', basePosition=[0.0,0.0,-0.1],
    hand_id = pybullet_instance.loadURDF(args['hand_path'] + '/' + this_hand, useFixedBase=True,
                         basePosition=[0.0, 0.0, 0.05], flags=pybullet_instance.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
    # print('object path',object_path)
    obj_id = pybullet_instance.loadURDF(object_path, basePosition=[0.0, 0.10, .0], flags=pybullet_instance.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)

    #obj_id = pybullet_instance.loadSoftBody("/home/ubuntu/Mojograsp/mojo-grasp/demos/rl_demo/resources/object_models/Jeremiah_Shapes/Shapes/torus.vtk", mass = 3, scale = 1, useNeoHookean = 1, NeoHookeanMu = 180, NeoHookeanLambda = 600, NeoHookeanDamping = 0.01, collisionMargin = 0.006, useSelfCollision = 1, frictionCoeff = 0.5, repulsionStiffness = 800)


    # Create TwoFingerGripper Object and set the initial joint positions
    hand = TwoFingerGripper(hand_id, path=args['hand_path'] + '/' + this_hand,hand_params=hand_param_dict)
    
    # change visual of gripper
    pybullet_instance.changeVisualShape(hand_id, -1, rgbaColor=[0.3, 0.3, 0.3, 1])
    pybullet_instance.changeVisualShape(hand_id, 0, rgbaColor=[1, 0.5, 0, 1])
    pybullet_instance.changeVisualShape(hand_id, 1, rgbaColor=[0.3, 0.3, 0.3, 1])
    pybullet_instance.changeVisualShape(hand_id, 3, rgbaColor=[1, 0.5, 0, 1])
    pybullet_instance.changeVisualShape(hand_id, 4, rgbaColor=[0.3, 0.3, 0.3, 1])
    pybullet_instance.changeVisualShape(plane_id,-1,rgbaColor=[1,1,1,1])
    obj = ObjectWithVelocity(obj_id, path=object_path,name='obj_2')

    # input('heh')
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
    if frictionList is None:
        pybullet_instance.changeDynamics(plane_id,-1,lateralFriction=0.05, spinningFriction=0.05, rollingFriction=0.05)
        pybullet_instance.changeDynamics(obj.id, -1, mass=.03, restitution=.95, lateralFriction=1)
    else:
        pybullet_instance.changeDynamics(hand_id, -1,lateralFriction=frictionList[0], spinningFriction=frictionList[1], rollingFriction=frictionList[2])
        pybullet_instance.changeDynamics(plane_id, -1,lateralFriction=frictionList[3], spinningFriction=frictionList[4], rollingFriction=frictionList[5])
        pybullet_instance.changeDynamics(obj.id, -1, mass=.03, restitution=.95, lateralFriction=frictionList[6],spinningFriction=frictionList[7], rollingFriction=frictionList[8])


    if contactList is not None:
        pybullet_instance.changeDynamics(hand_id, -1,contactStiffness=contactList[0],contactDamping=contactList[1], restitution=contactList[2])

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
        hand, obj, state, action, reward, env, start_holder, args=arg_dict, hand_type=hand_type)
    
    # data recording
    record_data = MultiprocessRecordData(rank,
        data_path=args['save_path'], state=state, action=action, reward=reward, save_all=False, controller=manipulation.controller)
    
    # gym wrapper around pybullet environment
    if args['model'] == 'PPO':
        gym_env = multiprocess_gym_wrapper.MultiprocessGymWrapper(env, manipulation, record_data, args)
    #elif 'DDPG' in args['model']:
        #gym_env = multiproccess_gym_wrapper_her.MultiprocessGymWrapper(env, manipulation, record_data, args)
    return gym_env, args, [pose_list,eval_pose_list]



def multiprocess_evaluate_loaded(filepath, shape_key=None, hand='A', ori=0, eval_set='full'):
    # load a trained model and test it on its test set
    import os
    demo_path = os.path.dirname(os.path.realpath(__file__))
    shape_dict = {'square':demo_path+"/resources/object_models/Jeremiah_Shapes/40x40_square.urdf",
        'square15':demo_path+"/resources/object_models/Jeremiah_Shapes/40x40_square_15.urdf",
        'square2':demo_path+"/resources/object_models/Jeremiah_Shapes/40x40_square_2.urdf",
        'square25':demo_path+"/resources/object_models/Jeremiah_Shapes/40x40_square_25.urdf",
        'square3':demo_path+"/resources/object_models/Jeremiah_Shapes/40x40_square_3.urdf",
        'circle':demo_path+"/resources/object_models/Jeremiah_Shapes/20_r_circle.urdf",
        'circle15':demo_path+"/resources/object_models/Jeremiah_Shapes/20_r_circle_15.urdf",
        'circle2':demo_path+"/resources/object_models/Jeremiah_Shapes/20_r_circle_2.urdf",
        'circle25':demo_path+"/resources/object_models/Jeremiah_Shapes/20_r_circle_25.urdf",
        'circle3':demo_path+"/resources/object_models/Jeremiah_Shapes/20_r_circle_3.urdf",
        'triangle':demo_path+"/resources/object_models/Jeremiah_Shapes/40x40_triangle.urdf",
        'triangle15':demo_path+"/resources/object_models/Jeremiah_Shapes/40x40_triangle_15.urdf",
        'triangle2':demo_path+"/resources/object_models/Jeremiah_Shapes/40x40_triangle_2.urdf",
        'triangle25':demo_path+"/resources/object_models/Jeremiah_Shapes/40x40_triangle_25.urdf",
        'triangle3':demo_path+"/resources/object_models/Jeremiah_Shapes/40x40_triangle_3.urdf",
        'teardrop':demo_path+"/resources/object_models/Jeremiah_Shapes/50x30_teardrop.urdf",
        'teardrop15':demo_path+"/resources/object_models/Jeremiah_Shapes/50x30_teardrop_15.urdf",
        'teardrop2':demo_path+"/resources/object_models/Jeremiah_Shapes/50x30_teardrop_2.urdf",
        'teardrop25':demo_path+"/resources/object_models/Jeremiah_Shapes/50x30_teardrop_25.urdf",
        'teardrop3':demo_path+"/resources/object_models/Jeremiah_Shapes/50x30_teardrop_3.urdf",
        'trapazoid':demo_path+"/resources/object_models/Jeremiah_Shapes/trapazoid.urdf",
        'pentagon':demo_path+"/resources/object_models/Jeremiah_Shapes/pentagon.urdf",
        'square_circle' :demo_path+"/resources/object_models/Jeremiah_Shapes/square_circle.urdf"
        }
    
    print('Evaluating on hands A or B')
    print('Hand A: 2v2_50.50_50.50_53')
    print('Hand B: 2v2_65.35_65.35_53')
    num_cpu = 16

    with open(filepath, 'r') as argfile:
        args = json.load(argfile)
    # args['eval-tsteps'] = 20
    high_level_folder = os.path.abspath(filepath)
    high_level_folder = os.path.dirname(high_level_folder)
    print(high_level_folder)
    key_file = os.path.abspath(__file__)
    key_file = os.path.dirname(key_file)
    key_file = os.path.join(key_file,'resources','hand_bank','hand_params.json')
    args['domain_randomization_finger_friction'] = False
    args['domain_randomization_floor_friction'] = False
    args['domain_randomization_object_mass'] = False
    args['domain_randomization_object_size'] = False
    args['finger_random_start'] = False
    args['random_shapes'] = False

    with open(key_file,'r') as hand_file:
        hand_params = json.load(hand_file)
    if args['model'] == 'PPO':
        model_type = PPO
    elif 'DDPG' in args['model']:
        model_type = DDPG
    elif 'TD3' in args['model']:
        model_type = TD3
    print('LOADING A MODEL')
    args['state_noise']=0.0
    # print('HARDCODING THE TEST PATH TO BE THE ROTATION TEST')
    # args['test_path'] ="/home/mothra/mojo-grasp/demos/rl_demo/resources/Big_rotation_15_test.csv"

    if not('contact_start' in args.keys()):
        args['contact_start'] = True
        print('we didnt have a contact start so we set it to true')
    if hand =='A':
        args['hand_file_list'] = ["2v2_50.50_50.50_1.1_53/hand/2v2_50.50_50.50_1.1_53.urdf"]
        ht = hand
    elif hand =='B':
        args['hand_file_list'] = ["2v2_65.35_65.35_1.1_53/hand/2v2_65.35_65.35_1.1_53.urdf"]
        ht = hand
    elif hand =='C':
        args['hand_file_list'] = ["2v2_65.35_50.50_1.1_53/hand/2v2_65.35_50.50_1.1_53.urdf"]
        ht = hand
    else:
        print('not going to evaluate, hand is wrong')
    if shape_key is None:
        print('no shape parameter provided, using the 1st shape in the config file')
        print('to evaluate multiple shapes, call the function multiple times with different shapes each time')
        args['object_path'] = [args['object_path'][0]]
        shape_key = 'Eval_'
    else:
        shape_path = shape_dict[shape_key]
        args['object_path'] = [shape_path]
    if 'Rotation' in args['task']:
        args['test_path'] = "./resources/Solo_rotation_test.csv"
    vec_env = SubprocVecEnv([make_env(args,[i,num_cpu],hand_info=hand_params) for i in range(num_cpu)])
    vec_env.env_method('set_reduced_save_type', False)
    # Change to nn.ReLu in kwargs
    model = model_type("MlpPolicy", vec_env, tensorboard_log=args['tname'], policy_kwargs={'log_std_init':-2.3}).load(args['save_path']+'best_model', env=vec_env)
    if 'Rotation' in args['task']:
        df = pd.read_csv('./resources/start_poses.csv', index_col=False)
        x_start = df['x']
        y_start = df['y']

        vec_env.env_method('evaluate', ht)
        for x,y in zip(x_start,y_start):
            vec_env.env_method('set_goal_holder_pos',[x,y])
            for _ in range(int(128/16)):
                # print('about to reset')
                obs = vec_env.reset()
                
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
        vec_env.env_method('evaluate', ht)
        vec_env.env_method('set_record_folder',shape_key+'_'+ht+'_'+str(ori), top_folder = 'Eval_Tests')

        if eval_set == 'single' or 'single_ori':
            x_start = [x_start[0]]
            y_start = [y_start[0]]

        for x, y in zip(x_start, y_start):
            tihng = {'goal_position': [x, y]}
            print('THING', tihng)

            if eval_set == 'ori':
                vec_env.env_method('set_reset_ori', tihng['goal_position'])
            elif eval_set == 'single_ori':
                vec_env.env_method('set_reset_single_ori', tihng['goal_position'], ori)
            else:
                vec_env.env_method('set_reset_point', tihng['goal_position'])

            for _ in range(int(1200 / 16)):
                obs = vec_env.reset()
                done = [False, False]
                while not all(done):
                    action, _ = model.predict(obs, deterministic=True)
                    vec_env.step_async(action)
                    obs, _, done, _ = vec_env.step_wait()

    vec_env.env_method('disconnect')

def asterisk_test(filepath,hand_type,frictionList = None, contactList = None, shape=None, iteration = None):

    # load a trained model and test it on its test set
    print('Evaluating on hands A or B')
    print('Hand A: 2v2_50.50_50.50_53')
    print('Hand B: 2v2_65.35_65.35_53')
    with open(filepath, 'r') as argfile:
        args = json.load(argfile)
    # args['eval-tsteps'] = 20
    high_level_folder = os.path.abspath(filepath)
    high_level_folder = os.path.dirname(high_level_folder)
    print(high_level_folder)
    key_file = os.path.abspath(__file__)
    key_file = os.path.dirname(key_file)
    key_file = os.path.join(key_file,'resources','hand_bank','hand_params.json')
    args['domain_randomization_finger_friction'] = False
    args['domain_randomization_floor_friction'] = False
    args['domain_randomization_object_mass'] = False
    args['domain_randomization_object_size'] = False
    args['finger_random_start'] = False
    args['object_random_start'] = False
    with open(key_file,'r') as hand_file:
        hand_params = json.load(hand_file)
    if args['model'] == 'PPO':
        model_type = PPO
    elif 'DDPG' in args['model']:
        model_type = DDPG
    elif 'TD3' in args['model']:
        model_type = TD3
    print('LOADING A MODEL')

    demo_path = os.path.dirname(os.path.realpath(__file__))

    shape_dict = {'square':demo_path+"/resources/object_models/Jeremiah_Shapes/40x40_square.urdf",
        'square15':demo_path+"/resources/object_models/Jeremiah_Shapes/40x40_square_15.urdf",
        'square2':demo_path+"/resources/object_models/Jeremiah_Shapes/40x40_square_2.urdf",
        'square25':demo_path+"/resources/object_models/Jeremiah_Shapes/40x40_square_25.urdf",
        'square3':demo_path+"/resources/object_models/Jeremiah_Shapes/40x40_square_3.urdf",
        'circle':demo_path+"/resources/object_models/Jeremiah_Shapes/20_r_circle.urdf",
        'circle15':demo_path+"/resources/object_models/Jeremiah_Shapes/20_r_circle_15.urdf",
        'circle2':demo_path+"/resources/object_models/Jeremiah_Shapes/20_r_circle_2.urdf",
        'circle25':demo_path+"/resources/object_models/Jeremiah_Shapes/20_r_circle_25.urdf",
        'circle3':demo_path+"/resources/object_models/Jeremiah_Shapes/20_r_circle_3.urdf",
        'triangle':demo_path+"/resources/object_models/Jeremiah_Shapes/40x40_triangle.urdf",
        'triangle15':demo_path+"/resources/object_models/Jeremiah_Shapes/40x40_triangle_15.urdf",
        'triangle2':demo_path+"/resources/object_models/Jeremiah_Shapes/40x40_triangle_2.urdf",
        'triangle25':demo_path+"/resources/object_models/Jeremiah_Shapes/40x40_triangle_25.urdf",
        'triangle3':demo_path+"/resources/object_models/Jeremiah_Shapes/40x40_triangle_3.urdf",
        'teardrop':demo_path+"/resources/object_models/Jeremiah_Shapes/50x30_teardrop.urdf",
        'teardrop15':demo_path+"/resources/object_models/Jeremiah_Shapes/50x30_teardrop_15.urdf",
        'teardrop2':demo_path+"/resources/object_models/Jeremiah_Shapes/50x30_teardrop_2.urdf",
        'teardrop25':demo_path+"/resources/object_models/Jeremiah_Shapes/50x30_teardrop_25.urdf",
        'teardrop3':demo_path+"/resources/object_models/Jeremiah_Shapes/50x30_teardrop_3.urdf",
        'trapazoid':demo_path+"/resources/object_models/Jeremiah_Shapes/trapazoid.urdf",
        'pentagon':demo_path+"/resources/object_models/Jeremiah_Shapes/pentagon.urdf",
        'square_circle' :demo_path+"/resources/object_models/Jeremiah_Shapes/square_circle.urdf"
        }

    # print('HARDCODING THE TEST PATH TO BE THE ROTATION TEST')
    # args['test_path'] ="/home/mothra/mojo-grasp/demos/rl_demo/resources/Solo_rotation_test.csv"
    shape_key = shape
    if shape_key is None:
        print('no shape parameter provided, using the 1st shape in the config file')
        print('to evaluate multiple shapes, call the function multiple times with different shapes each time')
        args['object_path'] = [args['object_path'][0]]
        shape_key = 'Ast_'
    else:
        shape_path = shape_dict[shape_key]
        args['object_path'] = [shape_path]

    if not('contact_start' in args.keys()):
        args['contact_start'] = True
        print('we didnt have a contact start so we set it to true')
    if hand_type =='A':
        args['hand_file_list'] = ["2v2_50.50_50.50_1.1_53/hand/2v2_50.50_50.50_1.1_53.urdf"]
    elif hand_type == 'B':
        args['hand_file_list'] = ["2v2_65.35_65.35_1.1_53/hand/2v2_65.35_65.35_1.1_53.urdf"]    
    elif hand_type =='C':
        args['hand_file_list'] = ["2v2_65.35_50.50_1.1_53/hand/2v2_65.35_50.50_1.1_53.urdf"]
    else:
        print('get fucked')
        assert 1==0
    asterisk_thing = [[0,0.07],[0.0495,0.0495],[0.07,0.0],[0.0495,-0.0495],[0.0,-0.07],[-0.0495,-0.0495],[-0.07,0.0],[-0.0495,0.0495]]
    scaled_points = [
    [0.0, 0.04666666666666667], [0.033, 0.033], [0.04666666666666667, 0.0], 
    [0.033, -0.033], [0.0, -0.04666666666666667], [-0.033, -0.033], 
    [-0.04666666666666667, 0.0], [-0.033, 0.033]]

    if 'Rotation' in args['task']:
        print('get fucked')
        assert 1==0

    import pybullet as p2
    eval_env , _, poses= make_pybullet(args,p2, [0,1], hand_params, viz=False)
    eval_env.evaluate()
    eval_env.set_record_folder('Ast_' + hand_type + '_' + shape + '_' + str(iteration), top_folder = 'Ast_Tests')
    eval_env.reduced_saving = False
    model = model_type("MlpPolicy", eval_env, tensorboard_log=args['tname'], policy_kwargs={'log_std_init':-2.3}).load(args['save_path']+'best_model', env=eval_env)
    eval_env.episode_type = 'asterisk'
    if frictionList is not None:
        for friction in range(len(frictionList)):
            for j in range(1,51):
                modifiedFrictionList = frictionList.copy()
                #print("Copied Friction list in wrapper is :", modifiedFrictionList)
                modifiedFrictionList[friction] = j * 0.1 * modifiedFrictionList[friction]
                #print("Modified in wrapper is :", modifiedFrictionList)
                eval_env.set_friction(modifiedFrictionList)
                for i in asterisk_thing:
                    eval_env.manipulation_phase.state.objects[-1].set_all_pose(i)
                    obs = eval_env.reset()
                    # input('go')
                    eval_env.manipulation_phase.state.objects[-1].set_all_pose(i)
                    done = False

                    while not done:
                        action, _ = model.predict(obs,deterministic=True)
                        obs, _, done, _ = eval_env.step(action,hand_type=hand_type)

    elif contactList is not None:
        for contact in range(len(contactList)):
            for j in range(0,101):
                modifiedContactList = contactList.copy()
                #print("Copied Contact list in wrapper is :", modifiedContactList)
                modifiedContactList[contact] = j * 0.1 * modifiedContactList[contact]
                #print("Modified in wrapper is :", modifiedContactList)
                eval_env.set_contact(modifiedContactList)
                for i in asterisk_thing:
                    eval_env.manipulation_phase.state.objects[-1].set_all_pose(i)
                    obs = eval_env.reset()
                    # input('go')
                    eval_env.manipulation_phase.state.objects[-1].set_all_pose(i)
                    done = False

                    while not done:
                        action, _ = model.predict(obs,deterministic=True)
                        obs, _, done, _ = eval_env.step(action,hand_type=hand_type)

    else:
        for i in asterisk_thing:
            eval_env.manipulation_phase.state.objects[-1].set_all_pose(i)
            obs = eval_env.reset()
            eval_env.manipulation_phase.state.objects[-1].set_all_pose(i)
            done = False
            #input('go')
            while not done:
                action, _ = model.predict(obs,deterministic=True)
                obs, _, done, _ = eval_env.step(action,hand_type=hand_type)

def rotation_test(filepath, hand_type):
    # load a trained model and test it on its test set
    print('Evaluating on hands A or B')
    print('Hand A: 2v2_50.50_50.50_53')
    print('Hand B: 2v2_65.35_65.35_53')
    with open(filepath, 'r') as argfile:
        args = json.load(argfile)
    # args['eval-tsteps'] = 20
    high_level_folder = os.path.abspath(filepath)
    high_level_folder = os.path.dirname(high_level_folder)
    print(high_level_folder)
    key_file = os.path.abspath(__file__)
    key_file = os.path.dirname(key_file)
    key_file = os.path.join(key_file,'resources','hand_bank','hand_params.json')
    args['domain_randomization_finger_friction'] = False
    args['domain_randomization_floor_friction'] = False
    args['domain_randomization_object_mass'] = False
    args['domain_randomization_object_size'] = False
    args['finger_random_start'] = False
    args['object_random_start'] = False
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
    max_ang = 50/180*np.pi
    args['contact_start'] = True
    if not('contact_start' in args.keys()):
        args['contact_start'] = True
        print('we didnt have a contact start so we set it to true')
    if hand_type =='A':
        args['hand_file_list'] = ["2v2_50.50_50.50_1.1_53/hand/2v2_50.50_50.50_1.1_53.urdf"]
        start_angles = [[-0.67,1.45,0.67,-1.45]
                        ,[-0.4, 0.66, 0.4, -0.66]
                        ,[-1.12, 2.01, 1.12, -2.01]
                        ,[-1.05,1.24,0.41,-1.39]
                        ,[-0.41,1.39,1.05,-1.24]]
        goal_poses = [[-0.0,0.0],[0.0,0.03],[0.0,-0.03],[0.04,0.0],[-0.04,0.0]]
        angles = [max_ang, -max_ang]
    elif hand_type == 'B':
        args['hand_file_list'] = ["2v2_65.35_65.35_1.1_53/hand/2v2_65.35_65.35_1.1_53.urdf"]
        start_angles = [[-0.67,1.45,0.67,-1.45]
                        ,[-0.4, 0.66, 0.4, -0.66]
                        ,[-1.12, 2.01, 1.12, -2.01]
                        ,[-1.05,1.24,0.41,-1.39]
                        ,[-0.41,1.39,1.05,-1.24]]
        goal_poses = [[-0.0,0.0],[0.0,0.03],[0.0,-0.03],[0.04,0.0],[-0.04,0.0]]
        angles = [max_ang, -max_ang]
    elif hand_type =='A_A':
        args['hand_file_list'] = ["2v2_50.50_50.50_1.1_53/hand/2v2_50.50_50.50_1.1_53.urdf"]
        start_angles = [[-0.67,1.45,0.67,-1.45]
                        ,[-0.4, 0.66, 0.4, -0.66]
                        ,[-1.12, 2.01, 1.12, -2.01]
                        ,[-1.05,1.24,0.41,-1.39]
                        ,[-0.41,1.39,1.05,-1.24]]
        goal_poses = [[-0.0,0.0],[0.0,0.03],[0.0,-0.03],[0.04,0.0],[-0.04,0.0]]
        angles = np.linspace(-50/180*np.pi,50/180*np.pi,128)
    elif hand_type == 'B_B':
        args['hand_file_list'] = ["2v2_65.35_65.35_1.1_53/hand/2v2_65.35_65.35_1.1_53.urdf"]
        start_angles = [[-0.4,1.5,0.4,-1.5],[-0.19,0.59,0.20,-0.59],[-0.51,2.09,0.51,-2.09],
                            [-0.76,1.37,0.03,-1.29],[-0.03,1.29,0.76,-1.37]]
        goal_poses = [[-0.0,0.0],[0.0,0.03],[0.0,-0.03],[0.04,0.0],[-0.04,0.0]]
        angles = np.linspace(-50/180*np.pi,50/180*np.pi,128)
    else:
        print('get fucked')
        assert 1==0
    
            
    if 'Rotation' not in args['task']:
        print('get fucked')
        assert 1==0
    fuckdis = np.array([-40, 40,-30,25,-15,25,-45,15,-20,50])/180*np.pi
    import pybullet as p2
    eval_env , _, poses= make_pybullet(args,p2, [0,1], hand_params, viz=False)
    eval_env.evaluate()
    eval_env.set_record_folder('Ast_'+hand_type)
    eval_env.reduced_saving = False
    model = model_type("MlpPolicy", eval_env, tensorboard_log=args['tname'], policy_kwargs={'log_std_init':-2.3}).load(args['save_path']+'best_model', env=eval_env)
    eval_env.episode_type = 'asterisk'
    i = 0
    print('we are about to begin testing')
    for start_pos in goal_poses:
        for go in angles:
            print(go)
            eval_env.manipulation_phase.state.objects[-1].set_all_pose(start_pos,go)
            s_dict = {'goal_position':start_pos}#, 'fingers':start_angs}
            obs = eval_env.reset(s_dict)
            eval_env.manipulation_phase.state.objects[-1].set_all_pose(start_pos,go)
            done = False
            i += 1
            # input('next')
            while not done:
                action, _ = model.predict(obs,deterministic=True)
                obs, _, done, _ = eval_env.step(action,hand_type=hand_type)
    p2.disconnect()

def full_test(filepath, hand_type):
    # load a trained model and test it on its test set
    print('Evaluating on hands A or B')
    print('Hand A: 2v2_50.50_50.50_53')
    print('Hand B: 2v2_65.35_65.35_53')
    with open(filepath, 'r') as argfile:
        args = json.load(argfile)
    # args['eval-tsteps'] = 20
    high_level_folder = os.path.abspath(filepath)
    high_level_folder = os.path.dirname(high_level_folder)
    print(high_level_folder)
    key_file = os.path.abspath(__file__)
    key_file = os.path.dirname(key_file)
    key_file = os.path.join(key_file,'resources','hand_bank','hand_params.json')
    args['domain_randomization_finger_friction'] = False
    args['domain_randomization_floor_friction'] = False
    args['domain_randomization_object_mass'] = False
    args['domain_randomization_object_size'] = False
    args['finger_random_start'] = False
    args['object_random_start'] = False
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
    if hand_type =='A':
        args['hand_file_list'] = ["2v2_50.50_50.50_1.1_53/hand/2v2_50.50_50.50_1.1_53.urdf"]
    elif hand_type == 'B':
        args['hand_file_list'] = ["2v2_65.35_65.35_1.1_53/hand/2v2_65.35_65.35_1.1_53.urdf"]    
    else:
        print('get fucked')
        assert 1==0
    asterisk_thing = np.array([[0,0.07],[0.0495,0.0495],[0.07,0.0],[0.0495,-0.0495],[0.0,-0.07],[-0.0495,-0.0495],[-0.07,0.0],[-0.0495,0.0495],
                      [0,0.07],[0.0495,0.0495],[0.07,0.0],[0.0495,-0.0495],[0.0,-0.07],[-0.0495,-0.0495],[-0.07,0.0],[-0.0495,0.0495]]) *4/7
    angles = [0.2618,0.2618,0.2618,0.2618,0.2618,0.2618,0.2618,0.2618,
              -0.2618,-0.2618,-0.2618,-0.2618,-0.2618,-0.2618,-0.2618,-0.2618]
    if 'Rotation' in args['task']:
        print('get fucked')
        assert 1==0

    import pybullet as p2
    # input(len(asterisk_thing))
    eval_env , _, poses= make_pybullet(args,p2, [0,1], hand_params, viz=False)
    eval_env.evaluate()
    eval_env.set_record_folder('Ast_'+hand_type)
    eval_env.reduced_saving = False
    model = model_type("MlpPolicy", eval_env, tensorboard_log=args['tname'], policy_kwargs={'log_std_init':-2.3}).load(args['save_path']+'best_model', env=eval_env)
    eval_env.episode_type = 'asterisk'
    for i, ang in zip(asterisk_thing,angles):
        eval_env.manipulation_phase.state.objects[-1].set_all_pose(i,ang)
        obs = eval_env.reset()
        eval_env.manipulation_phase.state.objects[-1].set_all_pose(i,ang)
        done = False
        while not done:
            action, _ = model.predict(obs,deterministic=True)
            obs, _, done, _ = eval_env.step(action,hand_type=hand_type)
    p2.disconnect()

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
    # eval_env.set_record_folder()
    eval_env.reduced_saving = False
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

def get_dynamic(shape, pose, orientation):
    """
    Computes the dynamic state of the object by applying a quaternion rotation 
    and translation (only in x and y) to the input shape.
    """
    x, y, z = pose
    quaternion = np.array(orientation)
    shape = np.hstack((shape, np.full((shape.shape[0], 1), 0.0)))

    rotation_matrix = R.from_quat(quaternion).as_matrix()

    shape = shape @ rotation_matrix.T


    #print(shape)
    shape[:, 0] += x
    shape[:, 1] += y
    shape[:, 2] += z
    # print('############################################')
    # print(shape)
    # print('############################################')
    # print(shape.flatten())
    #input('look at it?')

    return shape

# def get_dynamic(shape, pose, orientation):
#     """
#     Method that takes in the slice and the object pose and orientation and returns the dynamic state of the object
#     """
#     shape = np.hstack((shape, np.full((shape.shape[0], 1), 0.00)))
#     x, y, z = pose
#     a, b, c, w = orientation

#     # Normalize the quaternion to ensure proper rotation
#     norm = np.sqrt(a**2 + b**2 + c**2 + w**2)
#     a, b, c, w = a / norm, b / norm, c / norm, w / norm

#     # Construct the 3D rotation matrix from the quaternion
#     rotation_matrix = np.array([
#         [1 - 2 * (b**2 + c**2), 2 * (a * b - w * c),     2 * (a * c + w * b)],
#         [2 * (a * b + w * c),     1 - 2 * (a**2 + c**2), 2 * (b * c - w * a)],
#         [2 * (a * c - w * b),     2 * (b * c + w * a),   1 - 2 * (a**2 + b**2)]
#     ])

#     # Apply the rotation to the shape
#     shape = shape @ rotation_matrix.T

#     # Apply the translation
#     shape[:, 0] += x
#     shape[:, 1] += y
#     shape[:, 2] += z

#     return shape

        
def replay(argpath, episode_path, object_path):
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
    args['domain_randomization_finger_friction'] = False
    args['domain_randomization_floor_friction'] = False
    args['domain_randomization_object_mass'] = False
    args['domain_randomization_object_size'] = False
    args['finger_random_start'] = False    
    args['object_path'] = object_path
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
    args['hand_file_list'] = ["2v2_50.50_50.50_1.1_53/hand/2v2_50.50_50.50_1.1_53.urdf"]
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
        start_position = {'goal_position':obj_temp, 'fingers':temp}

        _ = eval_env.reset(start_position)

    else:
        start_position = {'goal_position':[0,0]} #, 'fingers':temp}
        _ = eval_env.reset(start_position)
    #print(data['timestep_list'][0]['state']['goal_pose'])
    #print(data['timestep_list'][0]['state']['obj_2'])
    temp = data['timestep_list'][0]['state']['goal_pose']['goal_position']
    angle = data['timestep_list'][0]['state']['goal_pose']['goal_orientation']
    
    visual_list = pg.get_slice(object_path)


    # for i in visual_list:
    #     eval_env.env.make_viz_point([i[0],i[1],0.0005])

    # df2 = pd.read_csv('./resources/start_poses.csv', index_col=False)
    # x_start = df2['x']
    # y_start = df2['y']

    # for xi,yi in zip(x_start,y_start):
    #     eval_env.env.make_viz_point([xi,yi+0.1,0.0005])
    # df = pd.read_csv('./resources/test_points_big.csv', index_col=False)
    # x = df['x']
    # y = df['y']
    # pts = [[xi,yi+0.1,0] for xi,yi in zip(x,y)]
    # eval_env.env.make_viz_point(pts)


    # input('look at it')
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
                    basePosition=[obj_temp[0]-0.0025,obj_temp[1]+0.1-0.0025,0.15],
                    baseOrientation =quat,
                    useMaximalCoordinates=True)
    
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
    #print('starting position', f1_poses[0],f2_poses[0], joint_angles[0])
    # input('start')
    joints = []
    print("INITIAL")
    #print(p2.getBasePositionAndOrientation(eval_env.env.obj.id))

    for i,act in enumerate(actions):
        # print('action vs mirrored:', actions[i],act)
        #print('joints in pkl file',joint_angles[i])
        print(f'step {i}')
        #print(act)
        eval_env.step(np.array(act),viz=True)
        step_num +=1
        #print('reward from pickle', data['timestep_list'][i]['reward'])
        pose, ori = p2.getBasePositionAndOrientation(eval_env.env.obj.id) 
        print('position', pose)
        x,y,z = pose
        visual_list_2 = get_dynamic(visual_list,pose,ori)
        print(max(visual_list_2[:,2]))
        p = []
        for i in visual_list_2:
            pt = eval_env.env.make_viz_point([i[0],i[1],i[2]])
            p.append(pt)
            #print(p)
        #input('clear?')
        for pt in p:
            eval_env.env.remove_viz_point(pt)
        #input('next step?')
        # time.sleep(0.5)
        # print(f'finger poses in pkl file, {f1_poses[i]}, {f2_poses[i]}')
        # print(data['timestep_list'][i]['action'])
        # input('next step?')
        # joints.append()
    p2.disconnect()

def cosine_annealing_with_restarts(initial_lr, min_lr=3e-5, t_initial=1e5, mult_factor=2.0):
    def schedule(progress):
        total_timesteps = progress * t_initial * (mult_factor ** math.floor(math.log(progress + 1, mult_factor)))
        cycle_length = t_initial * (mult_factor ** math.floor(math.log(progress + 1, mult_factor)))
        t_mod = total_timesteps % cycle_length
        cosine_decay = 0.5 * (1 + math.cos(math.pi * t_mod / cycle_length))
        return min_lr + (initial_lr - min_lr) * cosine_decay
    return schedule

def main(filepath = None,learn_type='run', num_cpu=16, j_test='base'):
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
        print('oh boy')
    elif 'TD3' in args['model']:
        model_type = TD3

    vec_env = SubprocVecEnv([make_env(args,[i,num_cpu],hand_info=hand_params) for i in range(num_cpu)])
    train_timesteps = int(args['evaluate']*(args['tsteps']+1)/num_cpu)
    callback = multiprocess_gym_wrapper.MultiEvaluateCallback(vec_env,n_eval_episodes=int(1200), eval_freq=train_timesteps, best_model_save_path=args['save_path'])

    if learn_type == 'transfer':
        model = model_type("MlpPolicy", vec_env, tensorboard_log=args['tname'], policy_kwargs={'log_std_init':-2.3}).load(args['load_path']+'best_model', env=vec_env,tensorboard_log=args['tname'],)
        print('LOADING A MODEL')
    elif learn_type == 'run':
        if 'DDPG' in args['model']:
            n_actions = vec_env.action_space.shape[0]
            noise_std = 0.2
            action_noise = NormalActionNoise(
                mean=np.zeros(n_actions), sigma=noise_std * np.ones(n_actions)
            )

            model = DDPG(
                        "MultiInputPolicy",
                        vec_env,
                        replay_buffer_class=HerReplayBuffer,
                        replay_buffer_kwargs=dict(
                            n_sampled_goal=4,
                            goal_selection_strategy="future",
                        ),
                        verbose=1,
                        buffer_size=int(1e6),
                        learning_rate=1e-3,
                        learning_starts=10000,
                        action_noise=action_noise,
                        gamma=0.95,
                        batch_size=256,
                        tensorboard_log=args['tname'])
            print('DDPG MODEL INITIALIZED')

        elif j_test == 'base':
            model = model_type("MlpPolicy", vec_env,tensorboard_log=args['tname'], policy_kwargs={'log_std_init':-0.69,'activation_fn': nn.ReLU,
                                                                                                   'net_arch':dict(pi=(64,64,64),vf=(64,64,64))})
            print('J_TEST MODEL INITIALIZED')
        elif j_test == 'warm_restart':
            model = PPO(
                "MlpPolicy",
                vec_env,
                learning_rate=cosine_annealing_with_restarts(initial_lr=3e-4, min_lr=3e-5, t_initial=1e5, mult_factor=2.0),
                tensorboard_log=args['tname'],
                policy_kwargs={'log_std_init': -0.69, 'activation_fn': nn.ReLU})
        else:
            # use ReLu in Kwargs and log_std_init = -.69 
            model = model_type("MlpPolicy", vec_env,tensorboard_log=args['tname'])

    try:
        print('starting the training using', get_device())
        model.learn(total_timesteps=args['epochs']*(args['tsteps']+1), callback=callback)
        filename = os.path.dirname(filepath)
        model.save(filename+'/last_model')
        merge_from_folder(args['save_path']+'Test/')

        # multiprocess_evaluate(model,vec_env)
    except KeyboardInterrupt:
        filename = os.path.dirname(filepath)
        model.save(filename+'/canceled_model')

if __name__ == '__main__':
    import csv
    replay('./data/Mothra_Rotation/FTP_S3/experiment_config.json','./data/Mothra_Rotation/FTP_S3/Full_Ast_B/Episode_37.pkl')
    # sub_names = ['FTP_S3','JA_S3']
    # top_names = ['Mothra_Rotation']#,'Jeremiah_Rotation','HPC_Rotation'] #['N_mothra_slide_rerun','N_HPC_slide_rerun','J_HPC_rerun'] # ,

    test_shape_list = ['square','square25','circle','circle25','triangle','triangle25','trapazoid','square_circle','pentagon']

    
    # angle = 0 #(in radians)
    # for item in test_shape_list:
    #     multiprocess_evaluate_loaded('./data/Full_90_2/SD_90_1/experiment_config.json',shape_key=item,hand="A", eval_set='single_ori', ori=angle)
    #     multiprocess_evaluate_loaded('./data/Full_90_2/SD_90_2/experiment_config.json',shape_key=item,hand="A", eval_set='single_ori', ori=angle)
    #     multiprocess_evaluate_loaded('./data/Full_90_2/SL_90_1/experiment_config.json',shape_key=item,hand="A", eval_set='single_ori', ori=angle)
    #     multiprocess_evaluate_loaded('./data/Full_90_2/SL_90_2/experiment_config.json',shape_key=item,hand="A", eval_set='single_ori', ori=angle)

    angle_list = [1.21538, 0.21243, 0.53392, 0.28959, -0.87162, -1.50185, 0.83918, -0.14781, -0.48328, -1.0089] # RADIANS
    # angle_list_2 = [round((num * 180 / np.pi),2) for num in angle_list]
    # print(sorted(angle_list_2))
    start_i = 6   # resume from i=6
    errors = []   # to record any failures

    for i, rad in enumerate(angle_list[start_i:], start=start_i):
        print(f"\n=== Starting iteration {i} (rad={rad}) ===")
        for shape in test_shape_list:
            for path in [
                './data/Full_Set_90/Static90/experiment_config.json',
                './data/Full_Set_90/Dynamic90/experiment_config.json',
                './data/Full_Set_90/Latent90/experiment_config.json',
                './data/Full_Set_90/SD90/experiment_config.json',
                './data/Full_Set_90/SL90/experiment_config.json',
                './data/New_Static_90/experiment_config.json',
                './data/Full_90_2/Dynamic_90_1/experiment_config.json',
                './data/Full_90_2/Dynamic_90_2/experiment_config.json',
                './data/Full_90_2/Static_90_1/experiment_config.json',
                './data/Full_90_2/Static_90_2/experiment_config.json',
                './data/Full_90_2/Latent_90_1/experiment_config.json',
                './data/Full_90_2/Latent_90_2/experiment_config.json',
                './data/Full_90_2/SD_90_1/experiment_config.json',
                './data/Full_90_2/SD_90_2/experiment_config.json',
                './data/Full_90_2/SL_90_1/experiment_config.json',
                './data/Full_90_2/SL_90_2/experiment_config.json',
                './data/Corrected_Static_Runs/Corrected_Static_90_1/experiment_config.json'
            ]:
                try:
                    asterisk_test(
                        filepath=path,
                        hand_type="A",
                        shape=shape,
                        iteration=i
                    )
                except Exception as e:
                    # record the failure but keep going
                    errors.append((i, rad, shape, path, str(e)))
                    print(f"  ✗ Error at i={i}, shape={shape}, path={path}: {e}")

        print(f"=== Finished iteration {i} ===")

    #multiprocess_evaluate_loaded('./data/New_Static_90/experiment_config.json', shape_key='square', eval_set='single_ori', ori=45)

    #main('./data/Corrected_Static_Runs/Corrected_Static_90_2/experiment_config.json', j_test='base')
    #replay('./data/Full_Domain_Test/Dynamic/experiment_config.json', './data/Full_Domain_Test/Dynamic/square_A/Episode_1180.pkl', '/home/ubuntu/Mojograsp/mojo-grasp/demos/rl_demo/resources/object_models/Jeremiah_Shapes/40x40_square.urdf')
    # replay('./data/NTestLayer/Dynamic/experiment_config.json', './data/NTestLayer/Dynamic/triangle_A/Episode_787.pkl')