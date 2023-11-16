#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 10:05:29 2023

@author: orochi
"""

# import gymnasium as gym
# from gymnasium import spaces
import gym
from gym import spaces
# from environment import Environment
import numpy as np
from mojograsp.simcore.state import State
from mojograsp.simcore.reward import Reward
import pybullet as p
from PIL import Image
from stable_baselines3.common.callbacks import EvalCallback

class EvaluateCallback(EvalCallback):

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self.eval_env.envs[0].evaluate()
            temp = super(EvaluateCallback,self)._on_step()
            self.eval_env.envs[0].train()
            t1 = self.eval_env.envs[0].manipulation_phase.controller.mags
            # print(f'during previous 1000 steps there were: {len(t1)} times the \
            #       finger tip motion was too high with an average magnitude of \
            #       {np.average(t1)} and a maximum of {max(t1)}')
            self.eval_env.envs[0].manipulation_phase.controller.mags=[]

            return temp
        else:
            return True

    
class NoiseAdder():
    def __init__(self,mins,maxes):
        self.mins = mins
        self.maxes = maxes
        
    def add_noise(self,x_tensor,noise_percent):
        """
        adds noise to an array based on desired noise percent
        :param x_tensor: - array to get noisey
        :param noise_percent: - how much noise to add
        """
        t1 = np.random.normal(0,noise_percent, size=len(x_tensor))
        # print('range',(self.maxes-self.mins))
        # print('raw noise', t1)
        # print('end noise',t1 * (self.maxes-self.mins))
        # print('xtensor', x_tensor)
        # print(np.array(x_tensor)> self.mins )
        # print((np.array(x_tensor)> self.mins))
        return x_tensor + t1 * (self.maxes-self.mins)/2



class GymWrapper(gym.Env):
    '''
    Example environment that follows gym interface to allow us to use openai gym learning algorithms with mojograsp
    '''
    
    def __init__(self, rl_env, manipulation_phase,record_data, args):
        super(GymWrapper,self).__init__()
        self.env = rl_env
        self.discrete = False
        if self.discrete:
            self.action_space = spaces.MultiDiscrete([3,3,3,3])
        else:
            self.action_space = spaces.Box(low=np.array([-1,-1,-1,-1]), high=np.array([1,1,1,1]))
        self.manipulation_phase = manipulation_phase
        self.observation_space = spaces.Box(np.array(args['state_mins']),np.array(args['state_maxes']))
        self.STATE_NOISE = args['state_noise']
        if self.STATE_NOISE > 0:
            self.noisey_boi = NoiseAdder(np.array(args['state_mins']), np.array(args['state_maxes']))
        self.PREV_VALS = args['pv']
        self.REWARD_TYPE = args['reward']
        self.state_list = args['state_list']
        self.CONTACT_SCALING = args['contact_scaling']
        self.DISTANCE_SCALING = args['distance_scaling'] 
        try:
            self.ROTATION_SCALING = args['rotation_scaling']
        except:
            self.ROTATION_SCALING = 0
        self.image_path = args['save_path'] + 'Videos/'
        self.record = record_data
        self.eval = False
        self.viz = False
        self.eval_run = 0
        self.timestep = 0
        self.first = True
        self.small_enough = args['epochs'] <= 100000
        self.episode_type = 'train'
        try:
            self.SUCCESS_REWARD = args['success_reward']
        except KeyError:
            self.SUCCESS_REWARD = 1
        self.SUCCESS_THRESHOLD = args['sr']/1000
        self.camera_view_matrix = p.computeViewMatrix((0.0,0.1,0.5),(0.0,0.1,0.005), (0.0,1,0.0))
        # self.camera_projection_matrix = p.computeProjectionMatrix(-0.1,0.1,-0.1,0.1,-0.1,0.1)
        self.camera_projection_matrix = p.computeProjectionMatrixFOV(60,4/3,0.1,0.9)
        
    def reset(self,special=None):
        if not self.first:
            
            if self.manipulation_phase.episode >= self.manipulation_phase.state.objects[-1].len:
                self.manipulation_phase.reset()
            self.manipulation_phase.next_phase()

        self.timestep=0
        self.first = False
        if self.eval:
            print('evaluating at eval run', self.eval_run)
            # print('fack',self.manipulation_phase.state.objects[-1].run_num)
            self.eval_run +=1
        if special is not None:
            self.env.reset_to_pos(special[0],special[1])
        else:
            self.env.reset()
        self.manipulation_phase.setup()
        
        state, _ = self.manipulation_phase.get_episode_info()
        # print('state and prev states')
        # print(state['goal_pose']['goal_pose'])
        # print(state['previous_state'][0]['f1_pos'],state['previous_state'][0]['f2_pos'])
        state = self.build_state(state)
        
        # print('Episode ',self.manipulation_phase.episode,' goal pose', self.manipulation_phase.goal_position)
        # print('fack',self.manipulation_phase.state.objects[-1].run_num)
        return state

    def step(self, action, mirror=False, viz=False):
        '''
        currently this does not use the action fed in in action, it uses the action applied to self.action in the sim manager

        Parameters
        ----------
        action : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        if self.discrete:
            action = action-1
            # print(action)
        self.manipulation_phase.gym_pre_step(action)
        self.manipulation_phase.execute_action(viz=viz)
        done = self.manipulation_phase.exit_condition()
        self.manipulation_phase.post_step()
        
        if self.eval or self.small_enough:
            self.record.record_timestep()
        # print('recorded timesteps')
        state, reward = self.manipulation_phase.get_episode_info()
        # print(state['obj_2'])
        info = {}
        if mirror:
            state = self.build_mirror_state(state)
        else:
            state = self.build_state(state)
        if self.STATE_NOISE > 0:
            state = self.noisey_boi.add_noise(state, self.STATE_NOISE)
        reward, done2 = self.build_reward(reward)
        # print('about to set state')
        # self.manipulation_phase.state.set_state()
        # print(reward)
        done = done | done2

        
        if done:
            # print('done, recording stuff')
            if self.eval or self.small_enough:
                self.record.record_episode(self.episode_type)
                if self.eval:
                    self.record.save_episode(self.episode_type, use_reward_name=True)
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
                        # print(state_container['previous_state'][i]['obj_2']['pose'][1])
                        state.extend(state_container['previous_state'][i]['obj_2']['pose'][1])
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
                    elif key == 'gp':
                        state.extend(state_container['previous_state'][i]['goal_pose']['goal_pose'])
                    else:
                        raise Exception('key does not match list of known keys')

        for key in self.state_list:
            if key == 'op':
                state.extend(state_container['obj_2']['pose'][0][0:2])
            elif key == 'oo':
                state.extend(state_container['obj_2']['pose'][1])
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
            elif key == 'gp':
                state.extend(state_container['goal_pose']['goal_pose'])
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
                        temp = state_container['previous_state'][i]['goal_pose']['goal_pose']
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
                temp = state_container['goal_pose']['goal_pose']
                state.extend([-temp[0],temp[1]])
            else:
                raise Exception('key does not match list of known keys')
        return state
    
    def build_reward(self, reward_container: Reward):
        """
        Method takes in a Reward object
        Extracts reward information from state_container and returns it as a float
        based on the reward structure contained in self.REWARD_TYPE

        :param state: :func:`~mojograsp.simcore.reward.Reward` object.
        :type state: :func:`~mojograsp.simcore.reward.Reward`
        """        
        done2 = False


        if self.REWARD_TYPE == 'Sparse':
            tstep_reward = -1 + 2*(reward_container['distance_to_goal'] < self.SUCCESS_THRESHOLD)
        elif self.REWARD_TYPE == 'Distance':
            tstep_reward = max(-reward_container['distance_to_goal'],-1)
        elif self.REWARD_TYPE == 'Distance + Finger':
            tstep_reward = max(-reward_container['distance_to_goal']*self.DISTANCE_SCALING - max(reward_container['f1_dist'],reward_container['f2_dist'])*self.CONTACT_SCALING,-1)
        elif self.REWARD_TYPE == 'Hinge Distance + Finger':
            tstep_reward = reward_container['distance_to_goal'] < self.SUCCESS_THRESHOLD + max(-reward_container['distance_to_goal'] - max(reward_container['f1_dist'],reward_container['f2_dist'])*self.CONTACT_SCALING,-1)
        elif self.REWARD_TYPE == 'Slope':
            tstep_reward = reward_container['slope_to_goal'] * self.DISTANCE_SCALING
        elif self.REWARD_TYPE == 'Slope + Finger':
            tstep_reward = max(reward_container['slope_to_goal'] * self.DISTANCE_SCALING  - max(reward_container['f1_dist'],reward_container['f2_dist'])*self.CONTACT_SCALING,-1)
        elif self.REWARD_TYPE == 'SmartDistance + Finger':
            ftemp = max(reward_container['f1_dist'],reward_container['f2_dist'])
            temp = -reward_container['distance_to_goal'] * (1 + 4*reward_container['plane_side'])
            # print(reward_container['plane_side'])
            tstep_reward = max(temp*self.DISTANCE_SCALING - ftemp*self.CONTACT_SCALING,-1)
        elif self.REWARD_TYPE == 'ScaledDistance + Finger':
            ftemp = max(reward_container['f1_dist'], reward_container['f2_dist']) * 100 # 100 here to make ftemp = -1 when at 1 cm
            temp = -reward_container['distance_to_goal']/reward_container['start_dist'] * (1 + 4*reward_container['plane_side'])
            # print(reward_container['plane_side'])
            tstep_reward = temp*self.DISTANCE_SCALING - ftemp*self.CONTACT_SCALING
        elif self.REWARD_TYPE == 'ScaledDistance+ScaledFinger':
            ftemp = -max(reward_container['f1_dist'], reward_container['f2_dist']) * 100 # 100 here to make ftemp = -1 when at 1 cm
            temp = -reward_container['distance_to_goal']/reward_container['start_dist'] # should scale this so that it is -1 at start 
            ftemp,temp = max(ftemp,-2), max(temp, -2)
            # print(ftemp,temp)
            tstep_reward = temp*self.DISTANCE_SCALING + ftemp*self.CONTACT_SCALING
        elif self.REWARD_TYPE == 'SFS':
            tstep_reward = reward_container['slope_to_goal'] * self.DISTANCE_SCALING - max(reward_container['f1_dist'],reward_container['f2_dist'])*self.CONTACT_SCALING
            if (reward_container['distance_to_goal'] < self.SUCCESS_THRESHOLD) & (np.linalg.norm(reward_container['object_velocity']) <= 0.05):
                tstep_reward += self.SUCCESS_REWARD
                done2 = True
        elif self.REWARD_TYPE == 'DFS':
            ftemp = max(reward_container['f1_dist'],reward_container['f2_dist'])
            # assert ftemp >= 0
            tstep_reward = -reward_container['distance_to_goal'] * self.DISTANCE_SCALING  - ftemp*self.CONTACT_SCALING
            if (reward_container['distance_to_goal'] < self.SUCCESS_THRESHOLD) & (np.linalg.norm(reward_container['object_velocity']) <= 0.05):
                tstep_reward += self.SUCCESS_REWARD
                done2 = True
        elif self.REWARD_TYPE == 'Rotation':
            temp = -reward_container['distance_to_goal']/reward_container['start_dist'] *  self.DISTANCE_SCALING
            temp2 = -reward_container['scaled_angle_distance'] * self.ROTATION_SCALING
        elif self.REWARD_TYPE == 'SmartDistance + SmartFinger':
            ftemp = max(reward_container['f1_dist'],reward_container['f2_dist'])
            if ftemp > 0.001:
                ftemp = ftemp*ftemp*1000
            temp = -reward_container['distance_to_goal'] * (1 + 4*reward_container['plane_side'])
            tstep_reward = max(temp*self.DISTANCE_SCALING - ftemp*self.CONTACT_SCALING,-1)
        else:
            raise Exception('reward type does not match list of known reward types')
        return float(tstep_reward), done2
    
    def render(self):
        pass
    
    def close(self):
        p.disconnect()
        
    def evaluate(self):
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
