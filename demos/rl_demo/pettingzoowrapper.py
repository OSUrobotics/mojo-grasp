#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import pettingzoo
# import gymnasium as gym
# from gymnasium import spaces
from typing import Tuple
from pettingzoo import AECEnv
from gym import spaces
import gym
# from environment import Environment
import numpy as np
from mojograsp.simcore.state import State
from mojograsp.simcore.reward import Reward
from PIL import Image
from stable_baselines3.common.callbacks import EvalCallback
from demos.rl_demo.rl_gym_wrapper import NoiseAdder
import mojograsp.simcore.reward_functions as rf
import time
from copy import copy


class WrapWrap(gym.Env):
    def __init__(self,wrapped):
        print('yo dog, i heard you like wrappers, so i wrapped your wrapper for the wrappers')
        self.subthing = wrapped
        # self.observation_space = spaces.Box(np.array([1]),np.array([1]))
    def reset(self):
        self.subthing.reset()
        return self.subthing.observe(self.subthing.agent_selection), {}  

    def step(self,action, viz=False,hand_type=None):
        self.subthing.step(action, viz,hand_type)
        return self.subthing.last()

    def render(self):
        pass
    
    def close(self):
        self.subthing.close()

    def evaluate(self):
        self.subthing.evaluate()

    def train(self):
        self.subthing.train()

class FullTaskWrapper(AECEnv):
    def __init__(self, HRL_env, manipulation_phase, record_data, args):
        super(FullTaskWrapper,self).__init__()
        self.env = HRL_env

        self.p = self.env.p
        # self.action_space = spaces.Box(low=np.array(args['actor_mins']), high=np.array(args['actor_maxes']))
        self.manipulation_phase = manipulation_phase
        self.observation_space = spaces.Box(np.array(args['state_mins']),np.array(args['state_maxes']))
        self.PREV_VALS = args['pv']
        self.REWARD_TYPE = args['reward']
        self.TASK = args['task']
        self.manager_state_list = args['state_list']
        self.worker_state_list = args['worker_state_list']
        self.CONTACT_SCALING = args['contact_scaling']
        self.DISTANCE_SCALING = args['distance_scaling'] 
        self.ROTATION_SCALING = args['rotation_scaling']
        self.image_path = args['save_path'] + 'Videos/'
        self.record = record_data
        self.eval = False
        self.viz = False
        self.eval_run = 0
        self.timestep = 0
        self.count = 0
        self.past_time = time.time()
        self.actions = {'manger':[],'worker':[]}
        self.first = True
        self.small_enough = args['epochs'] <= 500000
        self.episode_type = 'train'
        # self.horizon = 25
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
                       'ROTATION_SCALING':self.ROTATION_SCALING,
                       'SUCCESS_REWARD':self.SUCCESS_REWARD}
        self.possible_agents = ["manager", "worker"]
        self.agents = copy(self.possible_agents)
        self._action_spaces = {"manager":spaces.Box(low=np.array([-0.08,-0.08,-50/180*np.pi]), high=np.array([0.08,0.08,50/180*np.pi])),"worker": spaces.Box(low=np.array([-1,-1,-1,-1]), high=np.array([1,1,1,1]))}
        self._observation_spaces = {
            agent: spaces.Box(np.array(args['state_mins']),np.array(args['state_maxes'])) for agent in self.possible_agents
        }
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self.build_manager_reward = rf.triple_scaled_slide
        self.build_worker_reward = rf.worker

    def observe(self, agent):
        print('observin')
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """
        # observation is updated as soon as possible in step function
        return np.array(self.observations[agent])

    def reset(self):
        self.agents = self.possible_agents[:]

        """
        Our agent_selector utility allows easy cyclic stepping through the agents list.
        """
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        self.count += 1
        print(self.count)
        if not self.first:
            if self.manipulation_phase.episode >= self.manipulation_phase.state.objects[-1].len:
                self.manipulation_phase.reset()
            new_goal,fingerys = self.manipulation_phase.next_ep()
        self.timestep=0
        self.first = False
        self.env.reset()

        self.manipulation_phase.setup()
        if self.eval:
            self.eval_run +=1
        
        state_container, _ = self.manipulation_phase.get_episode_info()
        # print('state before reset')
        # print(state['goal_pose'])

        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {'manager':self.build_state(state_container, self.manager_state_list),'worker':self.build_state(state_container, self.worker_state_list)}
        self.observations = {'manager':self.build_state(state_container, self.manager_state_list),'worker':self.build_state(state_container, self.worker_state_list)}
        self.num_moves = 0

    def step(self, action, viz=False,hand_type=None):
        '''
        Parameters
        ----------
        action : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        agent = self.agent_selection

        self._cumulative_rewards[agent] = 0
        print('we steppin')
        # self.actions[self.agent_selection] = action
        # self.state[self.agent_selection] = action
        if self._agent_selector.is_last():
            self.manipulation_phase.gym_pre_step(action)
            self.manipulation_phase.execute_action(viz=viz)
            done = self.manipulation_phase.exit_condition()
            self.manipulation_phase.post_step()
            state_container, reward_container = self.manipulation_phase.get_episode_info()
            # At this point we have the updated state after taking a move
            # So now we need to fill the observations for the next timestep?
            # Thinking we definitely fill the managers observation
            # We can add the other one elsewhere
            # rewards for all agents are placed in the .rewards dictionary
            # Now that we have actually moved we can get rewards
            self.rewards['manager'],_ = self.build_manager_reward(reward_container, self.tholds)
            self.rewards['worker'],_ = self.build_worker_reward(reward_container, self.tholds)

            # The truncations dictionary must be updated for all players.
            self.truncations = {
                agent:done for agent in self.agents
            }

            # manager gets new state
            self.state['manager'] = state_container
            self.observations['manager'] = self.build_state(state_container, self.manager_state_list)
        else:
            self.manipulation_phase.set_goal(action)
            state_container = self.manipulation_phase.get_state()
            
            self.state['worker'] = state_container
            state = self.build_state(state_container, self.worker_state_list)
            # necessary so that observe() returns a reasonable observation at all times.
            
            # no rewards are allocated until both players give an action
            self._clear_rewards()
            self.observations['worker'] = state
        
        if self.eval or self.small_enough:
            self.record.record_timestep()

        if done:
            # print('done, recording stuff')
            if self.eval or self.small_enough:
                self.record.record_episode(self.episode_type)
                if self.eval:
                    self.record.save_episode(self.episode_type, hand_type=hand_type)
                else:
                    self.record.save_episode(self.episode_type)

        self.timestep +=1
    
    def render(self):
        pass
    
    def close(self):
        self.p.disconnect()

    def build_state(self, state_container: State, state_list):
        """
        Method takes in a State object 
        Extracts state information from state_container and returns it as a list based on
        current used states contained in self.state_list

        :param state: :func:`~mojograsp.simcore.phase.State` object.
        :type state: :func:`~mojograsp.simcore.phase.State`
        """
        print('building state')
        angle_keys = ["finger0_segment0_joint","finger0_segment1_joint","finger1_segment0_joint","finger1_segment1_joint"]
        state = []
        if self.PREV_VALS > 0:
            for i in range(self.PREV_VALS):
                for key in state_list:
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
                        state.extend(state_container['previous_state'][i]['goal_pose']['upper_goal_position'])
                    elif key == 'go':
                        state.append(state_container['previous_state'][i]['goal_pose']['upper_goal_orientation'])
                    elif key == 'gf':
                        state.extend(state_container['previous_state'][i]['goal_pose']['goal_finger'])
                    else:
                        raise Exception('key does not match list of known keys')

        for key in state_list:
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
                state.extend(state_container['goal_pose']['upper_goal_position'])
            elif key == 'go':
                state.append(state_container['goal_pose']['upper_goal_orientation'])
            elif key == 'gf':
                state.extend(state_container['goal_pose']['goal_finger'])
            else:
                raise Exception('key does not match list of known keys')
        return np.array(state)

    def evaluate(self, ht=None):
        # print('EVALUATE TRIGGERED')
        self.eval = True
        self.eval_run = 0
        self.manipulation_phase.state.evaluate()
        self.manipulation_phase.state.reset()
        self.manipulation_phase.state.objects[-1].run_num = 0
        self.manipulation_phase.eval = True
        self.record.clear()
        self.episode_type = 'test'
        self.hand_type = ht

    def train(self):
        self.eval = False
        self.manipulation_phase.eval = False
        self.manipulation_phase.state.train()
        self.manipulation_phase.state.reset()
        self.reset()
        self.episode_type = 'train'