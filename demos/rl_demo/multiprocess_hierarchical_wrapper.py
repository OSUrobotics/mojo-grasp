#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 10:53:29 2023

@author: orochi
"""

# import gymnasium as gym
# from gymnasium import spaces
from typing import Tuple
import gym
from gym import spaces
# from environment import Environment
import numpy as np
from mojograsp.simcore.state import State
from mojograsp.simcore.reward import Reward
from PIL import Image
from stable_baselines3.common.callbacks import EvalCallback
from demos.rl_demo.rl_gym_wrapper import NoiseAdder
import mojograsp.simcore.reward_functions as rf
import time

class MultiEvaluateCallback(EvalCallback):

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            print('evaluating')
            self.eval_env.env_method('evaluate')
            temp = super(MultiEvaluateCallback,self)._on_step()
            self.eval_env.env_method('train')
            return temp
        else:
            return True

class MultiprocessGymWrapper(gym.Env):
    '''
    Environment that follows gym interface to allow us to use openai gym learning algorithms with mojograsp
    '''
    
    def __init__(self, rl_env, manipulation_phase, record_data, args):
        super(MultiprocessGymWrapper,self).__init__()
        self.env = rl_env
        self.discrete = False
        self.p = self.env.p
        if self.discrete:
            self.action_space = spaces.MultiDiscrete([3,3,3,3])
        else:
            self.action_space = spaces.Box(low=np.array([-1,-1,-1,-1]), high=np.array([1,1,1,1]))
        self.manipulation_phase = manipulation_phase
        self.observation_space = spaces.Box(np.array(args['state_mins']),np.array(args['state_maxes']))
        self.STATE_NOISE = args['state_noise']
        if self.STATE_NOISE > 0:
            print('we are getting noisey')
            self.noisey_boi = NoiseAdder(np.array(args['state_mins']), np.array(args['state_maxes']))
        self.PREV_VALS = args['pv']
        self.REWARD_TYPE = args['reward']
        self.TASK = args['task']
        self.state_list = args['state_list']
        self.CONTACT_SCALING = args['contact_scaling']
        self.DISTANCE_SCALING = args['distance_scaling'] 
        self.image_path = args['save_path'] + 'Videos/'
        self.record = record_data
        self.eval = False
        self.viz = False
        self.eval_run = 0
        self.timestep = 0
        self.count = 0
        self.past_time = time.time()
        self.thing = []
        self.first = True
        self.small_enough = args['epochs'] <= 100000
        self.episode_type = 'train'
        try:
            self.SUCCESS_REWARD = args['success_reward']
        except KeyError:
            self.SUCCESS_REWARD = 1
        self.SUCCESS_THRESHOLD = args['sr']/1000
        self.camera_view_matrix = self.p.computeViewMatrix((0.0,0.1,0.5),(0.0,0.1,0.005), (0.0,1,0.0))
        # self.camera_projection_matrix = self.p = self.env.pp.computeProjectionMatrix(-0.1,0.1,-0.1,0.1,-0.1,0.1)
        self.camera_projection_matrix = self.p.computeProjectionMatrixFOV(60,4/3,0.1,0.9)

        self.build_reward = []
        self.prep_reward()
        self.tholds = {'SUCCESS_THRESHOLD':self.SUCCESS_THRESHOLD,
                       'DISTANCE_SCALING':self.DISTANCE_SCALING,
                       'CONTACT_SCALING':self.CONTACT_SCALING,
                       'SUCCESS_REWARD':self.SUCCESS_REWARD}
    def prep_reward(self):
        """
        Method takes in a Reward object
        Extracts reward information from state_container and returns it as a float
        based on the reward structure contained in self.REWARD_TYPE

        :param state: :func:`~mojograsp.simcore.reward.Reward` object.
        :type state: :func:`~mojograsp.simcore.reward.Reward`
        """        
        self.build_reward = rf.direction


    def reset(self,special=None):
        self.count += 1
        # if self.thing%1000 == 0:
        # print(self.thing)

        self.timestep=0
        self.first = False
        self.env.reset()
        self.manipulation_phase.setup()
        
        state, _ = self.manipulation_phase.get_episode_info()

        state = self.build_state(state)
        if self.count % 1000==0:
            print('thing')
        return state

    def step(self, action, mirror=False, viz=False,hand_type=None):
        '''
        Parameters
        ----------
        action : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        self.manipulation_phase.gym_pre_step(action)
        self.manipulation_phase.execute_action(viz=viz)
        done = self.manipulation_phase.exit_condition()
        self.manipulation_phase.post_step()
        
        if self.eval or self.small_enough:
            self.record.record_timestep()
        state, reward_container = self.manipulation_phase.get_episode_info()
        info = {}
        if mirror:
            state = self.build_mirror_state(state)
        else:
            state = self.build_state(state)
        if self.STATE_NOISE > 0:
            state = self.noisey_boi.add_noise(state, self.STATE_NOISE)
        # print(reward_container)
        reward, _ = self.build_reward(reward_container, self.tholds)

        if done:
            # print('done, recording stuff')
            if self.eval or self.small_enough:
                self.record.record_episode(self.episode_type)
                if self.eval:
                    self.record.save_episode(self.episode_type, hand_type=hand_type)
                else:
                    self.record.save_episode(self.episode_type)

        self.timestep +=1
        return state, reward, done, info
        
    
    def build_state(self, state_container: State):
        """
        Method takes in a State object 
        Extracts state information from state_container and returns it as a list based on
        current used states contained in self.state_list

        :param state: :func:`~mojograsp.simcore.phase.State` object.
        :type state: :func:`~mojograsp.simcore.phase.State`
        """
        angle_keys = ["finger0_segment0_joint","finger0_segment1_joint","finger1_segment0_joint","finger1_segment1_joint"]
        state = []
        if self.PREV_VALS > 0:
            for i in range(self.PREV_VALS):
                for key in self.state_list:
                    if key == 'op':
                        state.extend(state_container['previous_state'][i]['obj_2']['pose'][0][0:2])
                    elif key == 'oo':
                        state.extend(state_container['previous_state'][i]['obj_2']['pose'][1])
                    elif key == 'oa':
                        state.extend([np.sin(state_container['previous_state'][i]['obj_2']['z_angle']),np.cos(state_container['previous_state'][i]['obj_2']['z_angle'])])
                    elif key == 'ftp':
                        state.extend(state_container['previous_state'][i]['f1_pos'][0:2])
                        state.extend(state_container['previous_state'][i]['f2_pos'][0:2])
                    elif key == 'fbp':
                        state.extend(state_container['previous_state'][i]['f1_base'][0:2])
                        state.extend(state_container['previous_state'][i]['f2_base'][0:2])
                    elif key == 'fcp':
                        state.extend(state_container['previous_state'][i]['f1_contact_pos'][0:2])
                        state.extend(state_container['previous_state'][i]['f2_contact_pos'][0:2])
                    elif key == 'ja':
                        state.extend([state_container['previous_state'][i]['two_finger_gripper']['joint_angles'][item] for item in angle_keys])
                    elif key == 'fta':
                        state.extend([state_container['previous_state'][i]['f1_ang'],state_container['previous_state'][i]['f2_ang']])
                    elif key == 'eva':
                        state.extend(state_container['previous_state'][i]['two_finger_gripper']['eigenvalues'])
                    elif key == 'evc':
                        state.extend(state_container['previous_state'][i]['two_finger_gripper']['eigenvectors'])
                    elif key == 'evv':
                        evecs = state_container['previous_state'][i]['two_finger_gripper']['eigenvectors']
                        evals = state_container['previous_state'][i]['two_finger_gripper']['eigenvalues']
                        scaled = [evals[0]*evecs[0],evals[0]*evecs[2],evals[1]*evecs[1],evals[1]*evecs[3],
                                  evals[2]*evecs[4],evals[2]*evecs[6],evals[3]*evecs[5],evals[3]*evecs[7]]
                        state.extend(scaled)
                    elif key == 'params':
                        state.extend(state_container['hand_params'])
                    elif key == 'gp':
                        state.extend(state_container['previous_state'][i]['goal_pose']['goal_position'])
                    elif key == 'go':
                        state.append(state_container['previous_state'][i]['goal_pose']['goal_orientation'])
                    elif key == 'gf':
                        state.extend(state_container['previous_state'][i]['goal_pose']['goal_finger'])
                    else:
                        raise Exception('key does not match list of known keys')

        for key in self.state_list:
            if key == 'op':
                state.extend(state_container['obj_2']['pose'][0][0:2])
            elif key == 'oo':
                state.extend(state_container['obj_2']['pose'][1])
            elif key == 'oa':
                state.extend([np.sin(state_container['obj_2']['z_angle']),np.cos(state_container['obj_2']['z_angle'])])
            elif key == 'ftp':
                state.extend(state_container['f1_pos'][0:2])
                state.extend(state_container['f2_pos'][0:2])
            elif key == 'fbp':
                state.extend(state_container['f1_base'][0:2])
                state.extend(state_container['f2_base'][0:2])
            elif key == 'fcp':
                state.extend(state_container['f1_contact_pos'][0:2])
                state.extend(state_container['f2_contact_pos'][0:2])
            elif key == 'ja':
                state.extend([state_container['two_finger_gripper']['joint_angles'][item] for item in angle_keys])
            elif key == 'fta':
                state.extend([state_container['f1_ang'],state_container['f2_ang']])
            elif key == 'eva':
                state.extend(state_container['two_finger_gripper']['eigenvalues'])
            elif key == 'evc':
                state.extend(state_container['two_finger_gripper']['eigenvectors'])
            elif key == 'evv':
                evecs = state_container['two_finger_gripper']['eigenvectors']
                evals = state_container['two_finger_gripper']['eigenvalues']
                scaled = [evals[0]*evecs[0],evals[0]*evecs[2],evals[1]*evecs[1],evals[1]*evecs[3],
                          evals[2]*evecs[4],evals[2]*evecs[6],evals[3]*evecs[5],evals[3]*evecs[7]]
                state.extend(scaled)
            elif key == 'params':
                state.extend(state_container['hand_params'])
            elif key == 'gp':
                state.extend(state_container['goal_pose']['goal_position'])
            elif key == 'go':
                state.append(state_container['goal_pose']['goal_orientation'])
            elif key == 'gf':
                state.extend(state_container['goal_pose']['goal_finger'])
            else:
                raise Exception('key does not match list of known keys')
        return state

    def build_mirror_state(self, state_container: State):
        """
        Method takes in a State object 
        Extracts state information from state_container and returns it as a list based on
        current used states contained in self.state_list

        :param state: :func:`~mojograsp.simcore.phase.State` object.
        :type state: :func:`~mojograsp.simcore.phase.State`
        """
        angle_keys = ["finger0_segment0_joint","finger0_segment1_joint","finger1_segment0_joint","finger1_segment1_joint"]
        state = []
        if self.PREV_VALS > 0:
            for i in range(self.PREV_VALS):
                for key in self.state_list:
                    if key == 'op':
                        temp = state_container['previous_state'][i]['obj_2']['pose'][0]
                        state.extend([-temp[0],temp[1]])
                    elif key == 'ftp':
                        temp = state_container['previous_state'][i]['f2_pos'][0:2]
                        state.extend([-temp[0],temp[1]])
                        temp = state_container['previous_state'][i]['f1_pos'][0:2]
                        state.extend([-temp[0],temp[1]])
                    elif key == 'fbp':
                        temp = state_container['previous_state'][i]['f2_base']
                        state.extend([-temp[0],temp[1]])
                        temp = state_container['previous_state'][i]['f1_base']
                        state.extend([-temp[0],temp[1]])
                    elif key == 'fcp':
                        temp = state_container['previous_state'][i]['f2_contact_pos']
                        state.extend([-temp[0],temp[1]])
                        temp = state_container['previous_state'][i]['f1_contact_pos']
                        state.extend([-temp[0],temp[1]])
                    elif key == 'ja':
                        state.extend([-state_container['previous_state'][i]['two_finger_gripper']['joint_angles'][item] for item in angle_keys])
                    elif key == 'fta':
                        state.extend([-state_container['previous_state'][i]['f2_ang'],-state_container['previous_state'][i]['f1_ang']])
                    elif key == 'gp':
                        temp = state_container['previous_state'][i]['goal_pose']['goal_position']
                        state.extend([-temp[0],temp[1]])
                    else:
                        raise Exception('key does not match list of known keys')

        for key in self.state_list:
            if key == 'op':
                temp = state_container['obj_2']['pose'][0]
                state.extend([-temp[0],temp[1]])
            elif key == 'ftp':
                temp = state_container['f2_pos'][0:2]
                state.extend([-temp[0],temp[1]])
                temp = state_container['f1_pos'][0:2]
                state.extend([-temp[0],temp[1]])
            elif key == 'fbp':
                temp = state_container['f2_base']
                state.extend([-temp[0],temp[1]])
                temp = state_container['f1_base']
                state.extend([-temp[0],temp[1]])
            elif key == 'fcp':
                temp = state_container['f2_contact_pos']
                state.extend([-temp[0],temp[1]])
                temp = state_container['f1_contact_pos']
                state.extend([-temp[0],temp[1]])
            elif key == 'ja':
                state.extend([state_container['two_finger_gripper']['joint_angles'][item] for item in angle_keys])
            elif key == 'fta':
                state.extend([state_container['f2_ang'],state_container['f1_ang']])
            elif key == 'gp':
                temp = state_container['goal_pose']['goal_position']
                state.extend([-temp[0],temp[1]])
            else:
                raise Exception('key does not match list of known keys')
        return state
    
    def render(self):
        pass
    
    def close(self):
        self.p.disconnect()
        
    def evaluate(self):
        print('EVALUATE TRIGGERED')
        self.eval = True
        self.eval_run = 0
        self.manipulation_phase.state.evaluate()
        self.manipulation_phase.state.reset()
        self.manipulation_phase.state.objects[-1].run_num = 0
        self.manipulation_phase.eval = True
        self.record.clear()
        self.episode_type = 'test'
        
    def train(self):
        self.eval = False
        self.manipulation_phase.eval = False
        self.manipulation_phase.state.train()
        self.manipulation_phase.state.reset()
        self.reset()
        self.episode_type = 'train'


class SimpleHRLWrapper(gym.Env):
    def __init__(self, HRL_env, manipulation_phase, record_data, sub_policies, args):
        super(SimpleHRLWrapper,self).__init__()
        self.env = HRL_env
        self.sub_policies = sub_policies

        self.p = self.env.p
        self.action_space = spaces.Box(low=np.array([-1]* len(sub_policies)), high=np.array([1]* len(sub_policies)))
        self.manipulation_phase = manipulation_phase
        self.observation_space = spaces.Box(np.array(args['state_mins']),np.array(args['state_maxes']))
        self.STATE_NOISE = args['state_noise']
        if self.STATE_NOISE > 0:
            print('we are getting noisey')
            self.noisey_boi = NoiseAdder(np.array(args['state_mins']), np.array(args['state_maxes']))
        self.PREV_VALS = args['pv']
        self.REWARD_TYPE = args['reward']
        self.TASK = args['task']
        self.state_list = args['state_list']
        self.CONTACT_SCALING = args['contact_scaling']
        self.DISTANCE_SCALING = args['distance_scaling'] 
        self.image_path = args['save_path'] + 'Videos/'
        self.record = record_data
        self.eval = False
        self.viz = False
        self.eval_run = 0
        self.timestep = 0
        self.count = 0
        self.past_time = time.time()
        self.thing = []
        self.first = True
        self.small_enough = args['epochs'] <= 500000
        self.episode_type = 'train'
        try:
            self.SUCCESS_REWARD = args['success_reward']
        except KeyError:
            self.SUCCESS_REWARD = 1
        self.SUCCESS_THRESHOLD = args['sr']/1000
        self.camera_view_matrix = self.p.computeViewMatrix((0.0,0.1,0.5),(0.0,0.1,0.005), (0.0,1,0.0))
        # self.camera_projection_matrix = self.p = self.env.pp.computeProjectionMatrix(-0.1,0.1,-0.1,0.1,-0.1,0.1)
        self.camera_projection_matrix = self.p.computeProjectionMatrixFOV(60,4/3,0.1,0.9)

        self.build_reward = []
        self.tholds = {'SUCCESS_THRESHOLD':self.SUCCESS_THRESHOLD,
                       'DISTANCE_SCALING':self.DISTANCE_SCALING,
                       'CONTACT_SCALING':self.CONTACT_SCALING,
                       'SUCCESS_REWARD':self.SUCCESS_REWARD}
        
        self.build_reward = rf.double_scaled

    def step(self, action, mirror=False, viz=False,hand_type=None):
        '''
        Parameters
        ----------
        action : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        # action is a weight vector which we multiply by the output of the sub-polcies
        substate = self.manipulation_phase.get_built_sub_state(self.sub_state_list)
        sub_actions = [policy.predict(substate) for policy in self.sub_policies]
        final_action = np.dot(action,sub_actions)
        self.manipulation_phase.gym_pre_step(final_action)
        self.manipulation_phase.execute_action(viz=viz)
        done = self.manipulation_phase.exit_condition()
        self.manipulation_phase.post_step()
        
        if self.eval or self.small_enough:
            self.record.record_timestep()
        state, reward_container = self.manipulation_phase.get_episode_info()
        info = {}
        if mirror:
            state = self.build_mirror_state(state)
        else:
            state = self.build_state(state)
        if self.STATE_NOISE > 0:
            state = self.noisey_boi.add_noise(state, self.STATE_NOISE)
        # print(reward_container)
        reward, _ = self.build_reward(reward_container, self.tholds)

        if done:
            # print('done, recording stuff')
            if self.eval or self.small_enough:
                self.record.record_episode(self.episode_type)
                if self.eval:
                    self.record.save_episode(self.episode_type, hand_type=hand_type)
                else:
                    self.record.save_episode(self.episode_type)

        self.timestep +=1
        return state, reward, done, info

    def reset(self):
        self.count += 1

        self.timestep=0
        self.first = False
        self.env.reset()
        self.manipulation_phase.setup()
        
        state, _ = self.manipulation_phase.get_episode_info()

        state = self.build_state(state)
        if self.count % 1000==0:
            print('thing')
        return state
    
    def render(self):
        pass
    
    def close(self):
        self.p.disconnect()
        
    
    def build_state(self, state_container: State):
        """
        Method takes in a State object 
        Extracts state information from state_container and returns it as a list based on
        current used states contained in self.state_list

        :param state: :func:`~mojograsp.simcore.phase.State` object.
        :type state: :func:`~mojograsp.simcore.phase.State`
        """
        angle_keys = ["finger0_segment0_joint","finger0_segment1_joint","finger1_segment0_joint","finger1_segment1_joint"]
        state = []
        if self.PREV_VALS > 0:
            for i in range(self.PREV_VALS):
                for key in self.state_list:
                    if key == 'op':
                        state.extend(state_container['previous_state'][i]['obj_2']['pose'][0][0:2])
                    elif key == 'oo':
                        state.extend(state_container['previous_state'][i]['obj_2']['pose'][1])
                    elif key == 'oa':
                        state.extend([np.sin(state_container['previous_state'][i]['obj_2']['z_angle']),np.cos(state_container['previous_state'][i]['obj_2']['z_angle'])])
                    elif key == 'ftp':
                        state.extend(state_container['previous_state'][i]['f1_pos'][0:2])
                        state.extend(state_container['previous_state'][i]['f2_pos'][0:2])
                    elif key == 'fbp':
                        state.extend(state_container['previous_state'][i]['f1_base'][0:2])
                        state.extend(state_container['previous_state'][i]['f2_base'][0:2])
                    elif key == 'fcp':
                        state.extend(state_container['previous_state'][i]['f1_contact_pos'][0:2])
                        state.extend(state_container['previous_state'][i]['f2_contact_pos'][0:2])
                    elif key == 'ja':
                        state.extend([state_container['previous_state'][i]['two_finger_gripper']['joint_angles'][item] for item in angle_keys])
                    elif key == 'fta':
                        state.extend([state_container['previous_state'][i]['f1_ang'],state_container['previous_state'][i]['f2_ang']])
                    elif key == 'eva':
                        state.extend(state_container['previous_state'][i]['two_finger_gripper']['eigenvalues'])
                    elif key == 'evc':
                        state.extend(state_container['previous_state'][i]['two_finger_gripper']['eigenvectors'])
                    elif key == 'evv':
                        evecs = state_container['previous_state'][i]['two_finger_gripper']['eigenvectors']
                        evals = state_container['previous_state'][i]['two_finger_gripper']['eigenvalues']
                        scaled = [evals[0]*evecs[0],evals[0]*evecs[2],evals[1]*evecs[1],evals[1]*evecs[3],
                                  evals[2]*evecs[4],evals[2]*evecs[6],evals[3]*evecs[5],evals[3]*evecs[7]]
                        state.extend(scaled)
                    elif key == 'params':
                        state.extend(state_container['hand_params'])
                    elif key == 'gp':
                        state.extend(state_container['previous_state'][i]['goal_pose']['goal_position'])
                    elif key == 'go':
                        state.append(state_container['previous_state'][i]['goal_pose']['goal_orientation'])
                    elif key == 'gf':
                        state.extend(state_container['previous_state'][i]['goal_pose']['goal_finger'])
                    else:
                        raise Exception('key does not match list of known keys')

        for key in self.state_list:
            if key == 'op':
                state.extend(state_container['obj_2']['pose'][0][0:2])
            elif key == 'oo':
                state.extend(state_container['obj_2']['pose'][1])
            elif key == 'oa':
                state.extend([np.sin(state_container['obj_2']['z_angle']),np.cos(state_container['obj_2']['z_angle'])])
            elif key == 'ftp':
                state.extend(state_container['f1_pos'][0:2])
                state.extend(state_container['f2_pos'][0:2])
            elif key == 'fbp':
                state.extend(state_container['f1_base'][0:2])
                state.extend(state_container['f2_base'][0:2])
            elif key == 'fcp':
                state.extend(state_container['f1_contact_pos'][0:2])
                state.extend(state_container['f2_contact_pos'][0:2])
            elif key == 'ja':
                state.extend([state_container['two_finger_gripper']['joint_angles'][item] for item in angle_keys])
            elif key == 'fta':
                state.extend([state_container['f1_ang'],state_container['f2_ang']])
            elif key == 'eva':
                state.extend(state_container['two_finger_gripper']['eigenvalues'])
            elif key == 'evc':
                state.extend(state_container['two_finger_gripper']['eigenvectors'])
            elif key == 'evv':
                evecs = state_container['two_finger_gripper']['eigenvectors']
                evals = state_container['two_finger_gripper']['eigenvalues']
                scaled = [evals[0]*evecs[0],evals[0]*evecs[2],evals[1]*evecs[1],evals[1]*evecs[3],
                          evals[2]*evecs[4],evals[2]*evecs[6],evals[3]*evecs[5],evals[3]*evecs[7]]
                state.extend(scaled)
            elif key == 'params':
                state.extend(state_container['hand_params'])
            elif key == 'gp':
                state.extend(state_container['goal_pose']['goal_position'])
            elif key == 'go':
                state.append(state_container['goal_pose']['goal_orientation'])
            elif key == 'gf':
                state.extend(state_container['goal_pose']['goal_finger'])
            else:
                raise Exception('key does not match list of known keys')
        return state


class FeudalHRLWrapper(gym.Env):
    def __init__(self, HRL_env, manipulation_phase, record_data, sub_policy, args):
        super(FeudalHRLWrapper,self).__init__()
        self.env = HRL_env
        self.sub_policy = sub_policy

        self.p = self.env.p
        self.action_space = spaces.Box(low=np.array(args['actor_mins']), high=np.array(args['actor_maxes']))
        # self.action_space = spaces.Box(low=np.array([-1,-1,-1]), high=np.array([1,1,1]))
        self.manipulation_phase = manipulation_phase
        self.observation_space = spaces.Box(np.array(args['state_mins']),np.array(args['state_maxes']))
        self.STATE_NOISE = args['state_noise']
        if self.STATE_NOISE > 0:
            print('we are getting noisey')
            self.noisey_boi = NoiseAdder(np.array(args['state_mins']), np.array(args['state_maxes']))
        self.PREV_VALS = args['pv']
        self.REWARD_TYPE = args['reward']
        self.TASK = args['task']
        self.state_list = args['state_list']
        self.CONTACT_SCALING = args['contact_scaling']
        self.DISTANCE_SCALING = args['distance_scaling'] 
        self.image_path = args['save_path'] + 'Videos/'
        self.record = record_data
        self.eval = False
        self.viz = False
        self.eval_run = 0
        self.timestep = 0
        self.count = 0
        self.past_time = time.time()
        self.thing = []
        self.first = True
        self.small_enough = args['epochs'] <= 500000
        self.episode_type = 'train'
        try:
            self.SUCCESS_REWARD = args['success_reward']
        except KeyError:
            self.SUCCESS_REWARD = 1
        self.SUCCESS_THRESHOLD = args['sr']/1000
        self.camera_view_matrix = self.p.computeViewMatrix((0.0,0.1,0.5),(0.0,0.1,0.005), (0.0,1,0.0))
        # self.camera_projection_matrix = self.p = self.env.pp.computeProjectionMatrix(-0.1,0.1,-0.1,0.1,-0.1,0.1)
        self.camera_projection_matrix = self.p.computeProjectionMatrixFOV(60,4/3,0.1,0.9)

        self.build_reward = []
        self.tholds = {'SUCCESS_THRESHOLD':self.SUCCESS_THRESHOLD,
                       'DISTANCE_SCALING':self.DISTANCE_SCALING,
                       'CONTACT_SCALING':self.CONTACT_SCALING,
                       'SUCCESS_REWARD':self.SUCCESS_REWARD}
        
        self.build_reward = rf.double_scaled

    def step(self, action, mirror=False, viz=False,hand_type=None):
        '''
        Parameters
        ----------
        action : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        # print('got to step action')

        # action is a weight vector which we multiply by the output of the sub-polcies
        # print(action)
        self.manipulation_phase.set_goal(action)
        prev_state = self.manipulation_phase.get_state()
        # print('this one', prev_state['goal_pose'])
        # print('last one', prev_state['previous_state'][0]['goal_pose'])
        # print('further one', prev_state['previous_state'][1]['goal_pose'])
        # print('even more one', prev_state['previous_state'][2]['goal_pose'])
        # print('in step',type(prev_state))
        # We need to change the goal in prev state to match what we have here
        # prev_state = self.prep_substate(prev_state,action)
        # substate = self.sub_policy.build_state(prev_state)
        final_action,_ = self.sub_policy(prev_state)
        # print(final_action)
        self.manipulation_phase.gym_pre_step(final_action)
        self.manipulation_phase.execute_action(viz=viz)
        done = self.manipulation_phase.exit_condition()
        self.manipulation_phase.post_step()
        
        if self.eval or self.small_enough:
            self.record.record_timestep()
        state, reward_container = self.manipulation_phase.get_episode_info()
        info = {}
        if mirror:
            state = self.build_mirror_state(state)
        else:
            state = self.build_state(state)
        if self.STATE_NOISE > 0:
            state = self.noisey_boi.add_noise(state, self.STATE_NOISE)
        # print(reward_container)
        reward, _ = self.build_reward(reward_container, self.tholds)

        if done:
            # print('done, recording stuff')
            if self.eval or self.small_enough:
                self.record.record_episode(self.episode_type)
                if self.eval:
                    self.record.save_episode(self.episode_type, hand_type=hand_type)
                else:
                    self.record.save_episode(self.episode_type)

        self.timestep +=1
        return state, reward, done, info

    def reset(self):
        self.count += 1

        self.timestep=0
        self.first = False
        self.env.reset()
        self.manipulation_phase.setup()
        
        state, _ = self.manipulation_phase.get_episode_info()

        state = self.build_state(state)
        # print(state)
        if self.count % 1000==0:
            print('thing')
        return state
    
    def render(self):
        pass
    
    def close(self):
        self.p.disconnect()

    def build_state(self, state_container: State):
        """
        Method takes in a State object 
        Extracts state information from state_container and returns it as a list based on
        current used states contained in self.state_list

        :param state: :func:`~mojograsp.simcore.phase.State` object.
        :type state: :func:`~mojograsp.simcore.phase.State`
        """
        angle_keys = ["finger0_segment0_joint","finger0_segment1_joint","finger1_segment0_joint","finger1_segment1_joint"]
        state = []
        if self.PREV_VALS > 0:
            for i in range(self.PREV_VALS):
                for key in self.state_list:
                    if key == 'op':
                        state.extend(state_container['previous_state'][i]['obj_2']['pose'][0][0:2])
                    elif key == 'oo':
                        state.extend(state_container['previous_state'][i]['obj_2']['pose'][1])
                    elif key == 'oa':
                        state.extend([np.sin(state_container['previous_state'][i]['obj_2']['z_angle']),np.cos(state_container['previous_state'][i]['obj_2']['z_angle'])])
                    elif key == 'ftp':
                        state.extend(state_container['previous_state'][i]['f1_pos'][0:2])
                        state.extend(state_container['previous_state'][i]['f2_pos'][0:2])
                    elif key == 'fbp':
                        state.extend(state_container['previous_state'][i]['f1_base'][0:2])
                        state.extend(state_container['previous_state'][i]['f2_base'][0:2])
                    elif key == 'fcp':
                        state.extend(state_container['previous_state'][i]['f1_contact_pos'][0:2])
                        state.extend(state_container['previous_state'][i]['f2_contact_pos'][0:2])
                    elif key == 'ja':
                        state.extend([state_container['previous_state'][i]['two_finger_gripper']['joint_angles'][item] for item in angle_keys])
                    elif key == 'fta':
                        state.extend([state_container['previous_state'][i]['f1_ang'],state_container['previous_state'][i]['f2_ang']])
                    elif key == 'eva':
                        state.extend(state_container['previous_state'][i]['two_finger_gripper']['eigenvalues'])
                    elif key == 'evc':
                        state.extend(state_container['previous_state'][i]['two_finger_gripper']['eigenvectors'])
                    elif key == 'evv':
                        evecs = state_container['previous_state'][i]['two_finger_gripper']['eigenvectors']
                        evals = state_container['previous_state'][i]['two_finger_gripper']['eigenvalues']
                        scaled = [evals[0]*evecs[0],evals[0]*evecs[2],evals[1]*evecs[1],evals[1]*evecs[3],
                                  evals[2]*evecs[4],evals[2]*evecs[6],evals[3]*evecs[5],evals[3]*evecs[7]]
                        state.extend(scaled)
                    elif key == 'params':
                        state.extend(state_container['hand_params'])
                    elif key == 'gp':
                        state.extend(state_container['previous_state'][i]['goal_pose']['goal_position'])
                    elif key == 'go':
                        state.append(state_container['previous_state'][i]['goal_pose']['goal_orientation'])
                    elif key == 'gf':
                        state.extend(state_container['previous_state'][i]['goal_pose']['goal_finger'])
                    else:
                        raise Exception('key does not match list of known keys')

        for key in self.state_list:
            if key == 'op':
                state.extend(state_container['obj_2']['pose'][0][0:2])
            elif key == 'oo':
                state.extend(state_container['obj_2']['pose'][1])
            elif key == 'oa':
                state.extend([np.sin(state_container['obj_2']['z_angle']),np.cos(state_container['obj_2']['z_angle'])])
            elif key == 'ftp':
                state.extend(state_container['f1_pos'][0:2])
                state.extend(state_container['f2_pos'][0:2])
            elif key == 'fbp':
                state.extend(state_container['f1_base'][0:2])
                state.extend(state_container['f2_base'][0:2])
            elif key == 'fcp':
                state.extend(state_container['f1_contact_pos'][0:2])
                state.extend(state_container['f2_contact_pos'][0:2])
            elif key == 'ja':
                state.extend([state_container['two_finger_gripper']['joint_angles'][item] for item in angle_keys])
            elif key == 'fta':
                state.extend([state_container['f1_ang'],state_container['f2_ang']])
            elif key == 'eva':
                state.extend(state_container['two_finger_gripper']['eigenvalues'])
            elif key == 'evc':
                state.extend(state_container['two_finger_gripper']['eigenvectors'])
            elif key == 'evv':
                evecs = state_container['two_finger_gripper']['eigenvectors']
                evals = state_container['two_finger_gripper']['eigenvalues']
                scaled = [evals[0]*evecs[0],evals[0]*evecs[2],evals[1]*evecs[1],evals[1]*evecs[3],
                          evals[2]*evecs[4],evals[2]*evecs[6],evals[3]*evecs[5],evals[3]*evecs[7]]
                state.extend(scaled)
            elif key == 'params':
                state.extend(state_container['hand_params'])
            elif key == 'gp':
                state.extend(state_container['goal_pose']['goal_position'])
            elif key == 'go':
                state.append(state_container['goal_pose']['goal_orientation'])
            elif key == 'gf':
                state.extend(state_container['goal_pose']['goal_finger'])
            else:
                raise Exception('key does not match list of known keys')
        return state
