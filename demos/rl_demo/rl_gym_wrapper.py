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
        self.PREV_VALS = args['pv']
        self.REWARD_TYPE = args['reward']
        self.state_list = args['state_list']
        self.CONTACT_SCALING = args['contact_scaling']
        self.DISTANCE_SCALING = args['distance_scaling'] 
        self.record = record_data
        
    def reset(self):
        self.manipulation_phase.setup()
        self.env.reset()
        state, _ = self.manipulation_phase.get_episode_info()
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
        self.manipulation_phase.gym_pre_step(action)
        self.manipulation_phase.execute_action()
        self.env.step()
        done = self.manipulation_phase.exit_condition()
        self.manipulation_phase.post_step()
        self.record.record_timestep()
        state, reward = self.manipulation_phase.get_episode_info()
        info = {}
        state = self.build_state(state)
        reward = self.build_reward(reward)
        if done:
            print('done, recording stuff')
            self.record.record_episode()
            self.record.save_episode()
            self.manipulation_phase.next_phase()
            if self.manipulation_phase.episode >= 500:
                self.manipulation_phase.reset()
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