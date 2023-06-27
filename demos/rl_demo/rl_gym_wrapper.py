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

class NoiseAdder():
    def __init__(self,mins,maxes):
        self.mins = mins
        self.maxes = maxes
        
    def add_noise(self,x_tensor,noise_percent):
        """
        normalizes a numpy array to -1 and 1 using provided maximums and minimums
        :param x_tensor: - array to be normalized
        :param mins: - array containing minimum values for the parameters in x_tensor
        :param maxes: - array containing maximum values for the parameters in x_tensor
        """
        t1 = np.random.normal(0,noise_percent, size=len(x_tensor))
        print('range',(self.maxes-self.mins))
        print('raw noise', t1)
        print('end noise',t1 * (self.maxes-self.mins))
        print('xtensor', x_tensor)
        print(np.array(x_tensor)> self.mins )
        print((np.array(x_tensor)> self.mins))
        return x_tensor + t1 * (self.maxes-self.mins)/2
    
class GymWrapper(gym.Env):
    '''
    Example environment that follows gym interface to allow us to use openai gym learning algorithms with mojograsp
    '''
    
    def __init__(self, rl_env, manipulation_phase,record_data, args):
        super(GymWrapper,self).__init__()
        self.env = rl_env
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
        self.image_path = args['save_path'] + 'Videos/'
        self.record = record_data
        self.eval = False
        self.eval_names = None
        self.run_name = None
        self.eval_run = 0
        self.timestep = 0
        self.first = True
        self.camera_view_matrix = p.computeViewMatrix((0.0,0.1,0.5),(0.0,0.1,0.005), (0.0,1,0.0))
        # self.camera_projection_matrix = p.computeProjectionMatrix(-0.1,0.1,-0.1,0.1,-0.1,0.1)
        self.camera_projection_matrix = p.computeProjectionMatrixFOV(60,4/3,0.1,0.9)
        
    def reset(self):
        if not self.first:
            
            self.record.record_episode(self.eval)
            self.record.save_episode(self.eval, self.run_name)
            self.manipulation_phase.next_phase()
            if self.manipulation_phase.episode >= 500:
                self.manipulation_phase.reset()
        self.timestep=0
        self.first = False
        if self.eval:
            self.manipulation_phase.state.evaluate()
            self.run_name = self.eval_names[self.eval_run]
            self.eval_run +=1
        else:
            self.manipulation_phase.state.train()
        self.env.reset()
        self.manipulation_phase.setup()
        
        state, _ = self.manipulation_phase.get_episode_info()
        # print('state and prev states')
        # print(state['f1_pos'],state['f2_pos'])
        # print(state['previous_state'][0]['f1_pos'],state['previous_state'][0]['f2_pos'])
        state = self.build_state(state)
        
        print('Episode ',self.manipulation_phase.episode,' goal pose', self.manipulation_phase.goal_position)
        return state

    def step(self, action):
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
        # print('timestep num', self.timestep)
        self.manipulation_phase.gym_pre_step(action)
        self.manipulation_phase.execute_action()
        self.env.step()
        # print('just env stepped')
        done = self.manipulation_phase.exit_condition()
        self.manipulation_phase.post_step()
        
        self.record.record_timestep()
        # print('recorded timesteps')
        state, reward = self.manipulation_phase.get_episode_info()
        # print(state['obj_2'])
        info = {}
        state = self.build_state(state)
        if self.STATE_NOISE > 0:
            state = self.noisey_boi.add_noise(state, self.STATE_NOISE)
        reward = self.build_reward(reward)
        # print('about to set state')
        # self.manipulation_phase.state.set_state()
        # print(reward)
        
        if self.eval:
            
            img = p.getCameraImage(640, 480,viewMatrix=self.camera_view_matrix,
                                    projectionMatrix=self.camera_projection_matrix,
                                    shadow=1,
                                    lightDirection=[1, 1, 1])
            img = Image.fromarray(img[2])
            temp = self.run_name.split('.')[0]
            img.save(self.image_path+ temp + '_frame_'+ str(self.timestep)+'.png')
        
        if done:
            print('done, recording stuff')

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
        state = []
        if self.PREV_VALS > 0:
            for i in range(self.PREV_VALS):
                for key in self.state_list:
                    if key == 'op':
                        state.extend(state_container['previous_state'][i]['obj_2']['pose'][0][0:2])
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
                        state.extend([item for item in state_container['previous_state'][i]['two_finger_gripper']['joint_angles'].values()])
                    elif key == 'fta':
                        state.extend([state_container['previous_state'][i]['f1_ang'],state_container['previous_state'][i]['f2_ang']])
                    elif key == 'gp':
                        state.extend(state_container['previous_state'][i]['goal_pose']['goal_pose'])
                    else:
                        raise Exception('key does not match list of known keys')

        for key in self.state_list:
            if key == 'op':
                state.extend(state_container['obj_2']['pose'][0][0:2])
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
                state.extend([item for item in state_container['two_finger_gripper']['joint_angles'].values()])
            elif key == 'fta':
                state.extend([state_container['f1_ang'],state_container['f2_ang']])
            elif key == 'gp':
                state.extend(state_container['goal_pose']['goal_pose'])
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
        else:
            raise Exception('reward type does not match list of known reward types')
        return float(tstep_reward)
    
    def render(self):
        pass
    
    def close(self):
        pass