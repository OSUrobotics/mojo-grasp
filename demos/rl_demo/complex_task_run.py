#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 29 15:15:58 2023

@author: nigel swenson
"""

from pybullet_utils import bullet_client as bc
import pybullet_data
from demos.rl_demo import multiprocess_env
from demos.rl_demo import multiprocess_manipulation_phase
# import rl_env
from demos.rl_demo.multiprocess_state import MultiprocessState
from mojograsp.simcore.goal_holder import  GoalHolder, RandomGoalHolder, SingleGoalHolder
from demos.rl_demo import rl_action
from demos.rl_demo import multiprocess_reward
from demos.rl_demo import multiprocess_gym_wrapper
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import pandas as pd
from demos.rl_demo.multiprocess_record import MultiprocessRecordData
from mojograsp.simobjects.two_finger_gripper import TwoFingerGripper
from mojograsp.simobjects.object_with_velocity import ObjectWithVelocity
from mojograsp.simobjects.multiprocess_object import MultiprocessFixedObject
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

def make_pybullet(args, pybullet_instance, viz=True):

    
    if viz:
        physics_client = pybullet_instance.connect(pybullet_instance.GUI)
    else:
        physics_client = pybullet_instance.connect(pybullet_instance.DIRECT)
    pybullet_instance.setAdditionalSearchPath(pybullet_data.getDataPath())
    pybullet_instance.setGravity(0, 0, -10)
    pybullet_instance.setPhysicsEngineParameter(contactBreakingThreshold=.001)
    pybullet_instance.resetDebugVisualizerCamera(cameraDistance=.02, cameraYaw=0, cameraPitch=-89.9999,
                                 cameraTargetPosition=[0, 0.1, 0.5])
    if type(args['object_path']) == str:
        object_path = args['object_path']
        object_key = "small"
        print('older version of object loading, no object domain randomization used')
    else:
        object_path = args['object_path'][0%len(args['object_path'])]
        if 'add10' in object_path:
            object_key = 'add10'
        elif 'sub10' in object_path:
            object_key = 'sub10'
        else:
            object_key = 'small'
    # load objects into pybullet
    this_hand = "2v2_50.50_50.50_1.1_53/hand/2v2_50.50_50.50_1.1_53.urdf"
    hand_type = this_hand.split('/')[0]
    print(hand_type)
    key_file = './resources/hand_bank/hand_params.json'
    with open(key_file,'r') as hand_file:
        hand_info = json.load(hand_file)
    hand_keys = hand_type.split('_')
    info_1 = hand_info[hand_keys[-1]][hand_keys[1]]
    info_2 = hand_info[hand_keys[-1]][hand_keys[2]]
    hand_param_dict = {"link_lengths":[info_1['link_lengths'],info_2['link_lengths']],
                       "starting_angles":[info_1['contact_start_angles'][object_key][0],info_1['contact_start_angles'][object_key][1],-info_2['contact_start_angles'][object_key][0],-info_2['contact_start_angles'][object_key][1]],
                       "palm_width":info_1['palm_width'],
                       "hand_name":hand_type}
    # load objects into pybullet

    plane_id = pybullet_instance.loadURDF("plane.urdf", flags=pybullet_instance.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
    hand_id = pybullet_instance.loadURDF(args['hand_path'] + '/' + this_hand, useFixedBase=True,
                         basePosition=[0.0, 0.0, 0.05], flags=pybullet_instance.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
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
    
    # For standard loaded goal poses
    pose_list =[ [0.05,0]]
    orientations = [[0,0,0,0],[0,0,0,0]]
    eval_pose_list = [[0.05,0]]
    eval_orientations = [[0,0,0,0],[0,0,0,0]]
    goal_poses = SingleGoalHolder(pose_list)
    eval_goal_poses = SingleGoalHolder(eval_pose_list)
    # time.sleep(10)
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

    wall_id = pybullet_instance.loadURDF("./resources/object_models/wallthing/vertical_wall.urdf",basePosition=[0.0, 0.10, .05])
    cid = pybullet_instance.createConstraint(wall_id, -1, -1, -1, pybullet_instance.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0.09, 0.02], childFrameOrientation=[ 0, 0, 0.0, 1 ])
    wall = MultiprocessFixedObject(pybullet_instance,wall_id,"./resources/object_models/wallthing/vertical_wall.urdf",'wall')
    arg_dict = args.copy()
    if args['action'] == 'Joint Velocity':
        arg_dict['ik_flag'] = False
    else:
        arg_dict['ik_flag'] = True
    args['object_random_start'] = False

    # replay buffer
    replay_buffer = ReplayBufferPriority(buffer_size=4080000)
    
    # pybullet environment
    env = multiprocess_env.MultiprocessMazeEnv(pybullet_instance, hand=hand, obj=obj, wall=wall,hand_type=hand_type, goal_block=goal_poses,args=args)
    # Create phase
    manipulation = multiprocess_manipulation_phase.MultiprocessManipulation(
        hand, obj, state, action, reward, env, args=arg_dict, hand_type=hand_type)
    
    # data recording
    record_data = MultiprocessRecordData([0,1],
        data_path=args['save_path'], state=state, action=action, reward=reward, save_all=False, controller=manipulation.controller)
    # gym wrapper around pybullet environment
    gym_env = multiprocess_gym_wrapper.MultiprocessGymWrapper(env, manipulation, record_data, args)

    return gym_env, args, [obj_id, wall_id]


# control types: conventional + options, full, feudal? option keyboard?


def simple_interpolatinator(base_path):
    # drastically reduces number of points
    # find the vectors that the thing uses and only switch when there is a large change
    print(len(base_path))
    small_vectors = [[base_path[i+1][0] - base_path[i][0],base_path[i+1][1] - base_path[i][1],base_path[i+1][2] - base_path[i][2]] for i in range(len(base_path)-1)]
    small_vectors = [sm/np.linalg.norm(sm) for sm in small_vectors]
    ang_diff = [np.dot(small_vectors[i+1],small_vectors[i]) for i in range(len(small_vectors)-1)]
    turn_points = []
    for i, ag in enumerate(ang_diff):
        if ag <0.85:
            print('we found a change point')
            turn_points.append(base_path[i])
    turn_points.append(base_path[-1])
    return turn_points

def fancy_interpolatinator(base_path):
    # drastically reduces number of points into slide and rotation points
    # find the vectors that the thing uses and only switch when there is a large change
    print(len(base_path))
    small_vectors = [[base_path[i+1][0] - base_path[i][0],base_path[i+1][1] - base_path[i][1],base_path[i+1][2] - base_path[i][2]] for i in range(len(base_path)-1)]
    small_vectors = [sm/np.linalg.norm(sm) for sm in small_vectors]
    ang_diff = [np.dot(small_vectors[i+1],small_vectors[i]) for i in range(len(small_vectors)-1)]
    turn_points = []
    for i, ag in enumerate(ang_diff):
        if ag <0.9:
            print('we found a change point')
            turn_points.append(base_path[i])
    turn_points.append(base_path[-1])
    return turn_points

import sys
print(sys.path)
sys.path.append('/home/orochi/mojo/pybullet-planning')
import pybullet_tools.utils as pp

filepath = './data/HPC_slide_time_tests/20_contact/experiment_config.json'
filename = './data/HPC_slide_time_tests/20_contact'
with open(filepath, 'r') as argfile:
    args = json.load(argfile)
import pybullet as p
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
subpolicies = {}
names = ['best_model']
for name in names:
    model = model_type("MlpPolicy", None, _init_setup_model=False).load(filename+'/'+name)
    subpolicies[name] = model
env, args, ids = make_pybullet(args,p)
wall_id = ids[1]
obj_id = ids[0]
# print(self.p.getBaseVelocity(self.obj_id))
tihng = {'goal_position':[-0.05,0.0]}
state =env.reset(tihng)

goal_pose = (0.05,0.1,0)
input('LOOK AT IT')
obj_limits = ((-0.06, 0.06), (0.06,0.14))
obj_path = pp.plan_base_motion(obj_id, goal_pose, obj_limits, obstacles=[wall_id])
print('Original path length: ', len(obj_path))
# print(obj_path)
import matplotlib.pyplot as plt
xs = [o[0] for o in obj_path]
ys = [o[1] for o in obj_path]

reduced_obj_path = simple_interpolatinator(obj_path)
print(reduced_obj_path)

# state =env.reset(tihng)
count = 0
obj_temp = reduced_obj_path[0]
for goal in reduced_obj_path:
    print(goal)
    temp_id=p.loadURDF('./resources/object_models/2v2_mod/2v2_mod_cylinder_small_alt.urdf', flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES,
                globalScaling=0.2, basePosition=[goal[0],goal[1],0.11], baseOrientation=[ 0., 0, 0, 1 ])
    p.changeVisualShape(temp_id,-1, rgbaColor=[1, 0.0, 0.0, 1])
    constraint_id = p.createConstraint(temp_id, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], 
                                    [goal[0]-0.0025,goal[1]-0.0025,0.11], childFrameOrientation=[0,0,0,1])
    p.setCollisionFilterPair(temp_id, obj_id,-1,-1,0)
    print(temp_id)
plt.scatter(xs,ys)
plt.show()
# TODO get a policy that isnt shit working with this
# TODO get a gif of the thing workign with the wall in the way
# TODO maybe make a more diffcult environment for the thing
for obj_goal in reduced_obj_path:
    print('going to goal pose', obj_goal)
    env.set_goal([obj_goal[0],obj_goal[1]-0.1])
    # p.changeConstraint(constraint_id, [obj_goal[0]-0.0025,obj_goal[1]-0.0025,0.11])
    for i in range(10):
        action,_ = subpolicies['best_model'].predict(state,deterministic=True)
        print('dem actions', action)
        state, _, _, _ = env.step(np.array(action),viz=True)
        # time.sleep(0.4)
    print('finished goal number ', count)
    count +=1
