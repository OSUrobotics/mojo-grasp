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

def load_set(args):
    if args['points_path'] =='':
        x = [0.0]
        y = [0.0]
    else:
        df = pd.read_csv(args['points_path'], index_col=False)
        x = df['x']
        y = df['y']

    if 'test_path' in args.keys():
        df2 = pd.read_csv(args['test_path'],index_col=False)
        xeval = df2['x']
        yeval = df2['y']
    else:
        xeval = x.copy()
        yeval = y.copy()
    
    if 'rotation' in args['task']:
        orientations = np.random.uniform(-np.pi/2+0.1, np.pi/2-0.1,len(x))
        orientations = orientations + np.sign(orientations)*0.1
        eval_orientations = np.random.uniform(-np.pi/2+0.1, np.pi/2-0.1,len(xeval))
        eval_orientations = eval_orientations + np.sign(eval_orientations)*0.1
    else:
        orientations = np.zeros(len(x))
        eval_orientations = np.zeros(len(xeval))

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
    return pose_list, eval_pose_list, orientations, eval_orientations, finger_contacts, eval_finger_contacts

    
def make_pybullet(arg_dict, pybullet_instance, rank, hand_info, viz=False):
    # resource paths
    this_path = os.path.abspath(__file__)
    overall_path = os.path.dirname(os.path.dirname(os.path.dirname(this_path)))
    args=arg_dict
    print(args['task'])

    # load the desired test set based on the task
    pose_list, eval_pose_list, orientations, eval_orientations, finger_contacts, eval_finger_contacts = load_set(args)
    '''
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
        orientations = np.random.uniform(-np.pi/2+0.1, np.pi/2-0.1,1000)
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
    elif args['task'] == 'contact point':
        df = pd.read_csv(args['points_path'], index_col=False)
        x = df["x"]/2
        y = df["y"]/2
        print('x length', len(x))
        df2 = pd.read_csv(overall_path + '/demos/rl_demo/resources/test_points_big.csv', index_col=False)
        xeval = df2['x']/2
        yeval = df2['y']/2
        orientations = np.zeros(1000)
        eval_orientations = np.zeros(1200)

        finger_ys = np.random.uniform( 0.10778391676312778-0.02, 0.10778391676312778+0.02,(1000,2))
        finger_contacts = np.ones((1000,4))
        finger_contacts[:,0] = x + 0.026749999999999996
        finger_contacts[:,1] = y + finger_ys[:,0]
        finger_contacts[:,2] = x + -0.026749999999999996
        finger_contacts[:,3] = y + finger_ys[:,1]
        eval_finger_ys = np.random.uniform( 0.10778391676312778-0.02, 0.10778391676312778+0.02,(1200,2))
        eval_finger_contacts = np.ones((1200,4))
        eval_finger_contacts[:,0] = xeval + 0.026749999999999996
        eval_finger_contacts[:,1] = yeval + eval_finger_ys[:,0]
        eval_finger_contacts[:,2] = xeval + -0.026749999999999996
        eval_finger_contacts[:,3] = yeval + eval_finger_ys[:,1]
        eval_finger_contacts = np.array(eval_finger_contacts[int(num_eval*rank[0]/rank[1]):int(num_eval*(rank[0]+1)/rank[1])])
    else:
        raise KeyError('Task does not match known keys')
    asterisk_test_points = [[0,0.07],[0.0495,0.0495],[0.07,0.0],[0.0495,-0.0495],[0.0,-0.07],[-0.0495,-0.0495],[-0.07,0.0],[-0.0495,0.0495]]
    names = ['AsteriskSE.pkl','AsteriskS.pkl','AsteriskSW.pkl','AsteriskW.pkl','AsteriskNW.pkl','AsteriskN.pkl','AsteriskNE.pkl','AsteriskE.pkl']

    #uncomment for asterisk test
    # pose_list = asterisk_test_points
    # eval_pose_list = asterisk_test_points
    '''

    # Break test sets into pieces for multithreading
    num_eval = len(eval_pose_list)
    eval_pose_list = np.array(eval_pose_list[int(num_eval*rank[0]/rank[1]):int(num_eval*(rank[0]+1)/rank[1])])
    eval_orientations = np.array(eval_orientations[int(num_eval*rank[0]/rank[1]):int(num_eval*(rank[0]+1)/rank[1])])

    # set up goal holders based on task and points given
    if finger_contacts is not None:
        eval_finger_contacts = np.array(eval_finger_contacts[int(num_eval*rank[0]/rank[1]):int(num_eval*(rank[0]+1)/rank[1])])
        goal_poses = GoalHolder(pose_list,orientations,finger_contacts)
        eval_goal_poses = GoalHolder(eval_pose_list,eval_orientations,eval_finger_contacts)
    elif orientations is not None:
        goal_poses = GoalHolder(pose_list,orientations)
        eval_goal_poses = GoalHolder(eval_pose_list, eval_orientations)
    elif args['task'] == 'unplanned_random':
        goal_poses = RandomGoalHolder([0.02,0.065])
        eval_goal_poses = GoalHolder(eval_pose_list)
    else:    
        goal_poses = GoalHolder(pose_list)
        eval_goal_poses = GoalHolder(eval_pose_list)
    
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
    
    # pybullet environment
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
    
    actions = [a['action']['actor_output'] for a in data['timestep_list']]
    import pybullet as p2
    eval_env , _, poses= make_pybullet(args,p2, [0,1], hand_params,viz=True)
    eval_env.evaluate()

    # initialize with obeject in desired position. 
    # TODO fix this so that I don't need to comment/uncomment this to get desired behavior
    # start_position = {'goal_position':data['timestep_list'][0]['state']['goal_pose']['goal_position']}

    _ = eval_env.reset()
    print(data['timestep_list'][0]['state']['goal_pose'])
    temp = data['timestep_list'][0]['state']['goal_pose']['goal_pose']
    # p2.addUserDebugPoints([[data['timestep_list'][0]['state']['goal_pose']['goal_position'][0],data['timestep_list'][0]['state']['goal_pose']['goal_position'][1]+0.1,0.1]], [[1,1,1]], pointSize=5)
    visualShapeId = p2.createVisualShape(shapeType=p2.GEOM_SPHERE,
                                        rgbaColor=[1, 0, 0, 1],
                                        radius=0.005,
                                        specularColor=[0.4, .4, 0],
                                        visualFramePosition=[[temp[0],temp[1]+0.1,0.1]])
    collisionShapeId = p2.createCollisionShape(shapeType=p2.GEOM_SPHERE,
                                               radius=0.001)

    p2.createMultiBody(baseMass=0,
                    baseInertialFramePosition=[0,0,0],
                    baseCollisionShapeIndex=collisionShapeId,
                    baseVisualShapeIndex=visualShapeId,
                    basePosition=[temp[0]-0.0025,temp[1]+0.1-0.0025,0.11],
                    useMaximalCoordinates=True)
    
    p2.configureDebugVisualizer(p2.COV_ENABLE_RENDERING,1)
    step_num = 0
    for act in actions:
        eval_env.step(np.array(act),viz=True)
        print(f'step {step_num}')
        step_num +=1
            
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

        evaluate(filepath, "A")
        evaluate(filepath, "B")
    except KeyboardInterrupt:
        filename = os.path.dirname(filepath)
        model.save(filename+'/canceled_model')

if __name__ == '__main__':

    # main('./data/FTP_halfstate_A_rand_old_finger_poses/experiment_config.json','run')
    # main("./data/region_rotation_JA_finger/experiment_config.json",'run')
    # main("./data/JA_sliding_sub_policy/experiment_config.json",'run')
    # main("./data/FTP_halfstate_A_rand/experiment_config.json",'run')
    # evaluate("./data/FTP_halfstate_A_rand/experiment_config.json")
    # evaluate("./data/FTP_halfstate_A_rand/experiment_config.json","B")
    # evaluate("./data/FTP_fullstate_A_rand/experiment_config.json")
    # evaluate("./data/FTP_fullstate_A_rand/experiment_config.json","B")
    # evaluate("./data/JA_fullstate_A_rand/experiment_config.json")
    # evaluate("./data/JA_fullstate_A_rand/experiment_config.json","B")
    # replay("./data/JA_region_100_5/experiment_config.json","./data/JA_region_100_5/Eval_A/Episode_8.pkl")
    replay("./data/FTP_halfstate_A_rand/experiment_config.json","./data/FTP_halfstate_A_rand/Eval_B/Evaluate_72.pkl")