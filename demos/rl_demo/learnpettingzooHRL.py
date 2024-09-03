#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pybullet_data
from demos.rl_demo import pettingzoowrapper
from demos.rl_demo import multiprocess_env
from demos.rl_demo import multiprocess_manipulation_phase
# import rl_env
from demos.rl_demo.multiprocess_state import MultiprocessState
from mojograsp.simcore.goal_holder import  GoalHolder, RandomGoalHolder, SingleGoalHolder, HRLGoalHolder
from demos.rl_demo import rl_action
from demos.rl_demo import multiprocess_reward
from demos.rl_demo import multiproccess_gym_wrapper_her
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
import supersuit as ss

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
        goal_poses = HRLGoalHolder(pose_list, np.array(finger_starts[0:2]),orientations,finger_contacts, mix_orientation=True, mix_finger=True)
        eval_goal_poses = HRLGoalHolder(eval_pose_list, np.array(eval_finger_starts),eval_orientations,eval_finger_contacts)
    elif orientations is not None:
        print('we are shuffling the angle and fingertip for the training set with no finger goal')
        goal_poses = HRLGoalHolder(pose_list, np.array(finger_starts[0:2]), orientations,mix_orientation=True, mix_finger=True)
        eval_goal_poses = HRLGoalHolder(eval_pose_list, np.array(eval_finger_starts), eval_orientations)
    else:
        goal_poses = HRLGoalHolder(pose_list, finger_starts[0:2])
        eval_goal_poses = HRLGoalHolder(eval_pose_list, finger_starts[2:4])
    
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
            # print('older version of object loading, no object domain randomization used')
        else:
            # print('normally would get object DR but not this time')
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
        # print('STARTING AWAY FROM THE OBJECT')
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
    obj_id = pybullet_instance.loadURDF(object_path, basePosition=[0.0, 0.10, .05], flags=pybullet_instance.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
    # print(f'OBJECT ID:{obj_id}')

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

    gym_env = pettingzoowrapper.FullTaskWrapper(env, manipulation, record_data, args)

    return gym_env, args, [pose_list,eval_pose_list]
    #just for notes

def train(filepath, learn_type='run', num_cpu=16):
    # Create the vectorized environment
    print('cuda y/n?', get_device())

    with open(filepath, 'r') as argfile:
        args = json.load(argfile)

    key_file = os.path.abspath(__file__)
    key_file = os.path.dirname(key_file)
    key_file = os.path.join(key_file,'resources','hand_bank','hand_params.json')
    with open(key_file,'r') as hand_file:
        hand_params = json.load(hand_file)
    
    model_type = PPO
    import pybullet as pybullet_instance
    # from pettingzoo.utils.conversions import aec_to_parallel 
    from pantheonrl.envs.pettingzoo import PettingZooAECWrapper
    from pantheonrl.common.agents import OnPolicyAgent
    env, _ , _= make_pybullet(args, pybullet_instance, [0,1], hand_params, viz=False)
    # env = pettingzoowrapper.WrapWrap(env)
    # env = ss.pad_action_space_v0(env)
    env = PettingZooAECWrapper(env)
    partner = OnPolicyAgent(PPO('MlpPolicy', env.getDummyEnv(1), verbose=1,tensorboard_log=args['tname']+'/worker'),tensorboard_log=args['tname']+'/worker')

    # The second parameter ensures that the partner is assigned to a certain
    # player number. Forgetting this parameter would mean that all of the
    # partner agents can be picked as `player 2`, but none of them can be
    # picked as `player 3`.
    env.add_partner_agent(partner, player_num=1)
    train_timesteps = int(args['evaluate']*(args['tsteps']+1))
    worker_callback = pettingzoowrapper.WorkerEvaluateCallback(partner.model, args['save_path'])
    callback = pettingzoowrapper.ZooEvaluateCallback(env,n_eval_episodes=int(1200), eval_freq=int(train_timesteps), best_model_save_path=args['save_path'],callback_on_new_best=worker_callback)

    model = model_type("MlpPolicy", env,tensorboard_log=args['tname']+'/manager')
    try:
        model.learn(total_timesteps=args['epochs']*(args['tsteps']+1), callback=callback)
        filename = os.path.dirname(filepath)
        model.save(filename+'/manager_last_model')
        partner.model.save(filename+'/worker_last_model')
        merge_from_folder(args['save_path']+'Test/')

        # multiprocess_evaluate(model,vec_env)
    except KeyboardInterrupt:
        filename = os.path.dirname(filepath)
        model.save(filename+'/manager_canceled_model')
        partner.model.save(filename+'/worker_canceled_model')

def evaluate(filepath, modeltype='best'):
    with open(filepath, 'r') as argfile:
        args = json.load(argfile)
    filename = os.path.dirname(filepath)
    key_file = os.path.abspath(__file__)
    key_file = os.path.dirname(key_file)
    key_file = os.path.join(key_file,'resources','hand_bank','hand_params.json')
    with open(key_file,'r') as hand_file:
        hand_params = json.load(hand_file)
    
    model_type = PPO
    if modeltype =='best':
        manager_name = filename +'/manager_best_model'
        worker_name = filename +'/worker_best_model'
    elif modeltype =='last':
        manager_name = filename +'/manager_last_model'
        worker_name = filename +'/worker_last_model'
    elif modeltype =='canceled':
        manager_name = filename +'/manager_canceled_model'
        worker_name = filename +'/worker_canceled_model'
    import pybullet as pybullet_instance
    # print('hah')
    from pantheonrl.envs.pettingzoo import PettingZooAECWrapper
    from pantheonrl.common.agents import OnPolicyAgent
    env, _ , _= make_pybullet(args, pybullet_instance, [0,1], hand_params, viz=False)

    env = PettingZooAECWrapper(env)
    partner = OnPolicyAgent(PPO('MlpPolicy', env.getDummyEnv(1), verbose=1,tensorboard_log=args['tname']+'/worker'),tensorboard_log=args['tname']+'/worker')

    env.add_partner_agent(partner, player_num=1)

    model = model_type("MlpPolicy", env,tensorboard_log=args['tname']+'/manager')
    model.load(manager_name)
    partner.model.load(worker_name)
    env.base_env.evaluate('A')
    df = pd.read_csv('./resources/start_poses.csv', index_col=False)
    x_start = df['x']
    y_start = df['y']
    # input(len(x_start))
    print('YOOOO')
    for x,y in zip(x_start, y_start):
        tihng = {'goal_position':[x,y]}
        print('THING', tihng)
        env.base_env.set_reset_point(tihng['goal_position'])
        for i in range(1200):
            obs = env.reset()
            for _ in range(26):
                action, _ = model.predict(obs,deterministic=True)
                obs, _, done, _ = env.step(action,False)

def evaluate_loaded(filepath, modeltype='best'):
    with open(filepath, 'r') as argfile:
        args = json.load(argfile)
    filename = os.path.dirname(filepath)
    key_file = os.path.abspath(__file__)
    key_file = os.path.dirname(key_file)
    key_file = os.path.join(key_file,'resources','hand_bank','hand_params.json')
    with open(key_file,'r') as hand_file:
        hand_params = json.load(hand_file)
    
    model_type = PPO
    if modeltype =='best':
        manager_name = filename +'/manager_best_model'
        worker_name = filename +'/worker_best_model'
    elif modeltype =='last':
        manager_name = filename +'/manager_last_model'
        worker_name = filename +'/worker_last_model'
    elif modeltype =='canceled':
        manager_name = filename +'/manager_canceled_model'
        worker_name = filename +'/worker_canceled_model'
    import pybullet as pybullet_instance
    from pantheonrl.envs.pettingzoo import PettingZooAECWrapper
    from pantheonrl.common.agents import OnPolicyAgent
    env, _ , _= make_pybullet(args, pybullet_instance, [0,1], hand_params, viz=False)

    env = PettingZooAECWrapper(env)
    partner = OnPolicyAgent(PPO('MlpPolicy', env.getDummyEnv(1), verbose=1,tensorboard_log=args['tname']+'/worker'),tensorboard_log=args['tname']+'/worker')

    env.add_partner_agent(partner, player_num=1)

    model = model_type("MlpPolicy", env,tensorboard_log=args['tname']+'/manager')
    model.load(manager_name)
    partner.model.load(worker_name)
    env.base_env.evaluate()
    for i in range(1200):
        obs = env.reset()
        for _ in range(26):
            action, _ = model.predict(obs,deterministic=True)
            obs, _, done, _ = env.step(action,False)


def replay(configpath, replaypath):
    with open(configpath, 'r') as argfile:
        args = json.load(argfile)
    filename = os.path.dirname(configpath)
    key_file = os.path.abspath(__file__)
    key_file = os.path.dirname(key_file)
    key_file = os.path.join(key_file,'resources','hand_bank','hand_params.json')
    with open(key_file,'r') as hand_file:
        hand_params = json.load(hand_file)
    
    import pybullet as pybullet_instance
    from pantheonrl.envs.pettingzoo import PettingZooAECWrapper
    from pantheonrl.common.agents import OnPolicyAgent

    env, _ , _= make_pybullet(args, pybullet_instance, [0,1], hand_params, viz=True)
    env.evaluate()

    with open(replaypath,'rb') as playfile:
        data = pkl.load(playfile)
    
    data = data['timestep_list']
    actions = [i['action']['actor_output'] for i in data]
    start_point = data[0]['state']['obj_2']['pose'][0][0:2]
    start_point[1] = start_point[1]-0.1
    angdict = data[0]['state']['two_finger_gripper']['joint_angles']
    start_angs = [angdict['finger0_segment0_joint'],angdict['finger0_segment1_joint'],angdict['finger1_segment0_joint'],angdict['finger1_segment1_joint']]
    # print(actions)
    reset_dict = {'start_pos':start_point, 'finger_angs':start_angs}
    obs = env.reset(reset_dict)
    for action in actions:
        print(action)
        env.step(action,False)
        time.sleep(0.1)


if __name__ == '__main__':
    # train('./data/hrl_finger_action/experiment_config.json')

    # replay('./data/hrl_zoo_slide/experiment_config.json','./data/hrl_zoo_slide/Eval_A/Episode_50812.pkl')
    evaluate('./data/hrl_slide_limited_action/experiment_config.json',modeltype='best')