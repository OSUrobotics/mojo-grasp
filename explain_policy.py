import shap
import pybullet as p
import pybullet_data
from mojograsp.simcore.environment import EnvironmentDefault
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
from typing import Callable

def build_state_from_pkl(data, args):
    """
    Method takes in data loaded from pickle file object and config
    Extracts state information from state_container and returns it as a list based on
    current used states contained in self.state_list
    """
    angle_keys = ["finger0_segment0_joint","finger0_segment1_joint","finger1_segment0_joint","finger1_segment1_joint"]
    list_of_states = []
    data = data['timestep_list']
    for i in range(args['tsteps']):
        # print('doing the thing')
        state = []
        if args['pv'] > 0:
            for j in range(1,args['pv']+1):
                for key in args['state_list']:
                    if key == 'op':
                        state.extend(data[max(i-j,0)]['state']['obj_2']['pose'][0][0:2])
                    elif key == 'oo':
                        state.extend(data[max(i-j,0)]['state']['obj_2']['pose'][1])
                    elif key == 'ftp':
                        state.extend(data[max(i-j,0)]['state']['f1_pos'][0:2])
                        state.extend(data[max(i-j,0)]['state']['f2_pos'][0:2])
                    elif key == 'fbp':
                        state.extend(data[max(i-j,0)]['state']['f1_base'][0:2])
                        state.extend(data[max(i-j,0)]['state']['f2_base'][0:2])
                    elif key == 'fcp':
                        state.extend(data[max(i-j,0)]['state']['f1_contact_pos'][0:2])
                        state.extend(data[max(i-j,0)]['state']['f2_contact_pos'][0:2])
                    elif key == 'ja':
                        state.extend([data[max(i-j,0)]['state']['two_finger_gripper']['joint_angles'][item] for item in angle_keys])
                    elif key == 'fta':
                        state.extend([data[max(i-j,0)]['state']['f1_ang'],data[max(i-j,0)]['state']['f2_ang']])
                    elif key == 'eva':
                        state.extend(data[max(i-j,0)]['state']['two_finger_gripper']['eigenvalues'])
                    elif key == 'evc':
                        state.extend(data[max(i-j,0)]['state']['two_finger_gripper']['eigenvectors'])
                    elif key == 'evv':
                        evecs = data[max(i-j,0)]['state']['two_finger_gripper']['eigenvectors']
                        evals = data[max(i-j,0)]['state']['two_finger_gripper']['eigenvalues']
                        scaled = [evals[0]*evecs[0],evals[0]*evecs[2],evals[1]*evecs[1],evals[1]*evecs[3],
                                    evals[2]*evecs[4],evals[2]*evecs[6],evals[3]*evecs[5],evals[3]*evecs[7]]
                        state.extend(scaled)
                    elif key == 'gp':
                        state.extend(data[max(i-j,0)]['state']['goal_pose']['goal_pose'])
                    else:
                        raise Exception('key does not match list of known keys')

        for key in args['state_list']:
            if key == 'op':
                state.extend(data[i]['state']['obj_2']['pose'][0][0:2])
            elif key == 'oo':
                state.extend(data[i]['state']['obj_2']['pose'][1])
            elif key == 'ftp':
                state.extend(data[i]['state']['f1_pos'][0:2])
                state.extend(data[i]['state']['f2_pos'][0:2])
            elif key == 'fbp':
                state.extend(data[i]['state']['f1_base'][0:2])
                state.extend(data[i]['state']['f2_base'][0:2])
            elif key == 'fcp':
                state.extend(data[i]['state']['f1_contact_pos'][0:2])
                state.extend(data[i]['state']['f2_contact_pos'][0:2])
            elif key == 'ja':
                state.extend([data[i]['state']['two_finger_gripper']['joint_angles'][item] for item in angle_keys])
            elif key == 'fta':
                state.extend([data[i]['state']['f1_ang'],data[i]['state']['f2_ang']])
            elif key == 'eva':
                state.extend(data[i]['state']['two_finger_gripper']['eigenvalues'])
            elif key == 'evc':
                state.extend(data[i]['state']['two_finger_gripper']['eigenvectors'])
            elif key == 'evv':
                evecs = data[i]['state']['two_finger_gripper']['eigenvectors']
                evals = data[i]['state']['two_finger_gripper']['eigenvalues']
                scaled = [evals[0]*evecs[0],evals[0]*evecs[2],evals[1]*evecs[1],evals[1]*evecs[3],
                            evals[2]*evecs[4],evals[2]*evecs[6],evals[3]*evecs[5],evals[3]*evecs[7]]
                state.extend(scaled)
            elif key == 'gp':
                state.extend(data[i]['state']['goal_pose']['goal_pose'])
            else:
                raise Exception('key does not match list of known keys')
        list_of_states.append(state)
    return list_of_states


def explain_policy(filepath):
    '''
    Uses shapely values to explain which state values are most important. expects a filepath to the folder containing experiment_config.json
    with a trained model present and a full set of evaluations in filepath + Eval/
    '''
    


    with open(filepath+'experiment_config.json', 'r') as argfile:
        args = json.load(argfile)
    

    eval_files = os.listdir(filepath + 'Eval')
    state_info = []
    print('Eval file length',len(eval_files))
    for filename in eval_files:
        # print(filename)
        with open(filepath + 'Eval/'+filename, 'rb') as file:
            data = pkl.load(file)
            state_list = build_state_from_pkl(data, args)
            state_info.extend(state_list)
    physics_client = p.connect(p.DIRECT)
    hand_id = p.loadURDF(args['hand_path'], useFixedBase=True,
                         basePosition=[0.0, 0.0, 0.05], flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
    obj_id = p.loadURDF(args['object_path'], basePosition=[0.0, 0.10, .05], flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)

    x = [0]
    y = [0]
    pose_list = [x[0],y[0]]
    eval_pose_list = pose_list.copy()
    hand = IKGripper(hand_id, path=args['hand_path'])

    obj = ObjectWithVelocity(obj_id, path=args['object_path'],name='obj_2')

    goal_poses = GoalHolder(pose_list)
    eval_goal_poses = GoalHolder(eval_pose_list)

    state = StateRL(objects=[hand, obj, goal_poses], prev_len=args['pv'],eval_goals = eval_goal_poses)
    
    if args['freq'] ==240:
        action = rl_action.ExpertAction()
    else:
        action = rl_action.InterpAction(args['freq'])
    
    reward = rl_reward.ExpertReward()

    arg_dict = args.copy()
    if args['action'] == 'Joint Velocity':
        arg_dict['ik_flag'] = False
    else:
        arg_dict['ik_flag'] = True
        
    # replay buffer
    replay_buffer = ReplayBufferPriority(buffer_size=4080000)
    
    # environment and recording
    # env = rl_env.ExpertEnv(hand=hand, obj=obj, hand_type=arg_dict['hand'], rand_start=args['rstart'])

    env = EnvironmentDefault()
    # env = rl_env.ExpertEnv(hand=hand, obj=cylinder)
    
    # Create phase
    manipulation = manipulation_phase_rl.ManipulationRL(
        hand, obj, x, y, state, action, reward, env, replay_buffer=replay_buffer, args=arg_dict)
    
    
    # data recording
    record_data = RecordDataRLPKL(
        data_path=args['save_path'], state=state, action=action, reward=reward, save_all=False, controller=manipulation.controller)
    
    
    gym_env = rl_gym_wrapper.GymWrapper(env, manipulation, record_data, args)
    
    model = PPO("MlpPolicy", gym_env, tensorboard_log=args['tname'], policy_kwargs={'log_std_init':-2.3}).load(args['save_path']+'best_model')


    def F(state):
        out = np.zeros([4,len(state)])
        for i,v in enumerate(state):
            temp, _ = model.predict(v)
            out[:,i] = temp
        # print(out.shape)
        return out[0,:].flatten()
    

    state_info = np.array(state_info) # this should be a 30000x50 or so array
    print(state_info.shape)
    explainer = shap.KernelExplainer(F, state_info)
    # test = explainer(state_info[0])
    shap_values = explainer.shap_values(state_info[0:500,:], nsamples=50)
    shap.summary_plot(shap_values, state_info[0:500,:])

    return explainer, state_info


explainer, state_info = explain_policy('/home/mothra/mojo-grasp/demos/rl_demo/data/wedge_double/wedge_l-r/')
