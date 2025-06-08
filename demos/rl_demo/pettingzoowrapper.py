#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import pettingzoo
# import gymnasium as gym
# from gymnasium import spaces
from typing import Tuple
from gymnasium.spaces.space import Space
from pettingzoo import AECEnv
from gym import spaces
import gym
# from environment import Environment
import numpy as np
from mojograsp.simcore.state import State
from mojograsp.simcore.reward import Reward
from PIL import Image
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from demos.rl_demo.rl_gym_wrapper import NoiseAdder
import mojograsp.simcore.reward_functions as rf
import time
from copy import copy
from pettingzoo.utils import agent_selector
import os

class ZooEvaluateCallback(EvalCallback):
    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self.eval_env.envs[0].evaluate()
            temp = super(ZooEvaluateCallback,self)._on_step()
            print('about to train')
            # a = time.time()
            self.eval_env.envs[0].train()
            # print('finishing the evaluate callback, returning to training', time.time()-a)
            # self.eval_env.envs[0].manipulation_phase.controller.mags=[]

            return temp
        else:
            return True

class WorkerEvaluateCallback(BaseCallback):
    def __init__(self, worker_model, save_path):
        super().__init__()
        self.worker_model = worker_model
        self.model_save_path = save_path
    def _on_step(self) -> bool:
        self.worker_model.save(os.path.join(self.model_save_path, "worker_best_model"))
        return True

class FullTaskWrapper(AECEnv):
    def __init__(self, HRL_env, manipulation_phase, record_data, args):
        super(FullTaskWrapper,self).__init__()
        self.env = HRL_env
        self.eval_point = None

        self.p = self.env.p
        # self.action_space = spaces.Box(low=np.array(args['actor_mins']), high=np.array(args['actor_maxes']))
        self.manipulation_phase = manipulation_phase
        
        self.PREV_VALS = args['pv']
        self.REWARD_TYPE = args['reward']
        self.TASK = args['task']
        self.manager_state_list = args['manager_state_list']
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
        self.metadata = None
        self.render_mode = None
        # 
        self.DOMAIN_RANDOMIZATION_MASS = args['domain_randomization_object_mass']
        self.DOMAIN_RANDOMIZATION_FINGER = args['domain_randomization_finger_friction']
        self.DOMAIN_RANDOMIZATION_FLOOR = args['domain_randomization_floor_friction']
        # self.horizon = 25
        # self.max_num_agents = 2
        try:
            self.SUCCESS_REWARD = args['success_reward']
        except KeyError:
            self.SUCCESS_REWARD = 1
        self.SUCCESS_THRESHOLD = args['sr']/1000
        self.camera_view_matrix = self.p.computeViewMatrix((0.0,0.1,0.5),(0.0,0.1,0.005), (0.0,1,0.0))
        # self.camera_projection_matrix = self.p = self.env.pp.computeProjectionMatrix(-0.1,0.1,-0.1,0.1,-0.1,0.1)
        self.camera_projection_matrix = self.p.computeProjectionMatrixFOV(60,4/3,0.1,0.9)
        self.randomize_pose = args['object_random_start']
        self.randomize_fingers = args['finger_random_start']
        self.build_reward = []
        try:
            self.cirriculum = args['cirriculum']
        except:
            self.cirriculum = False
        self.tholds = {'SUCCESS_THRESHOLD':self.SUCCESS_THRESHOLD,
                       'DISTANCE_SCALING':self.DISTANCE_SCALING,
                       'CONTACT_SCALING':self.CONTACT_SCALING,
                       'ROTATION_SCALING':self.ROTATION_SCALING,
                       'SUCCESS_REWARD':self.SUCCESS_REWARD}
        self.possible_agents = ["manager", "worker"]
        self.agents = copy(self.possible_agents)
        # testing this with both at -1 and 1, this is actual thing:spaces.Box(low=np.array([-0.08,-0.08,-50/180*np.pi]), high=np.array([0.08,0.08,50/180*np.pi]))
        # print(args['manager_maxes'],args['manager_mins'])
        self._action_spaces = {"manager":spaces.Box(low=np.array(len(args['manager_mins'])*[-1]), high=np.array(len(args['manager_mins'])*[1])),
                               "worker": spaces.Box(low=np.array([-1,-1,-1,-1]), high=np.array([1,1,1,1]))}
        self.manager_normalizer = {'mins':np.array(args['manager_mins']),'diff':np.array(args['manager_maxes'])-np.array(args['manager_mins'])}
        if 'manager_state_maxes' in args.keys():
            self._observation_spaces = {
                # "manager": spaces.Box(
                #                 low=0, high=255, 
                #                 shape=(240, 240,1), 
                #                 dtype=np.uint8
                #             ),
                'manager': spaces.Box(np.array(args['manager_state_mins']),np.array(args['manager_state_maxes'])),
                'worker':  spaces.Box(np.array(args['worker_state_mins']),np.array(args['worker_state_maxes']))
            }
        else:
            self._observation_spaces = {
                agent: spaces.Box(np.array(args['state_mins']),np.array(args['state_maxes'])) for agent in self.possible_agents
            }
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        # KEEP IN MIND THIS IS ALWAYS HERE
        if "Rotation" in self.TASK:
            self.build_manager_reward = rf.manager_rotation
            if args['manager_action_dim'] == 3:
                self.build_worker_reward = rf.worker_object_pose
            else:
                self.build_worker_reward = rf.worker_object_pose_finger_rotation
        elif "Multi" in self.TASK:
            self.build_manager_reward = rf.sparse_multigoal
            self.build_worker_reward = rf.worker_multigoal
        elif self.TASK == "big_random":
            self.build_manager_reward = rf.manager
            self.build_worker_reward = rf.worker_object_position

        # self.count_test = 0

    def observation_space(self, agent) -> Space:
        return self._observation_spaces[agent]
    
    def observe(self, agent):
        # print('observin')
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """
        # observation is updated as soon as possible in step function
        # print(len(np.array(self.observations[agent])), self._observation_spaces)
        return np.array(self.observations[agent])
    
    def action_space(self, agent):
        return self._action_spaces[agent]
    
    def reset(self,reset_dict=None):
        self.agents = self.possible_agents[:]

        """
        Our agent_selector utility allows easy cyclic stepping through the agents list.
        """
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        self.count += 1
        if self.count%100 ==0:
            print(self.count)
        if not self.first:
            if self.manipulation_phase.episode >= self.manipulation_phase.state.objects[-1].len:
                self.manipulation_phase.reset()
            new_goal,fingerys = self.manipulation_phase.next_ep()
        else:
            new_goal = {'goal_position':[0,0]}
            fingerys = [0,0]

        self.env.apply_domain_randomization(self.DOMAIN_RANDOMIZATION_FINGER,self.DOMAIN_RANDOMIZATION_FLOOR,self.DOMAIN_RANDOMIZATION_MASS)


        self.timestep=0
        self.first = False
        
        if reset_dict is None:
            if self.eval_point is not None:
                self.env.reset(self.eval_point)
            elif self.randomize_pose:            
                random_start = np.random.uniform(0,1,2)
                x = (1-random_start[0]**2) * np.sin(random_start[1]*2*np.pi) * 0.06
                y = (1-random_start[0]**2) * np.cos(random_start[1]*2*np.pi) * 0.04

                if self.randomize_fingers:
                    self.env.reset([x,y],fingerys=fingerys)
                else:
                    self.env.reset([x,y])
            elif self.randomize_fingers:
                self.env.reset(fingerys=fingerys)
            else:
                self.env.reset()
        else:
            self.env.reset(reset_dict['start_pos'], finger=reset_dict['finger_angs'])
        
        self.manipulation_phase.setup()
        if self.eval:
            self.eval_run +=1
        
        state_container, _ = self.manipulation_phase.get_episode_info()
        # print('resetting', state_container['goal_pose'])
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {'manager':self.build_state(state_container, self.manager_state_list),'worker':self.build_state(state_container, self.worker_state_list)}
        self.observations = {'manager':self.build_state(state_container, self.manager_state_list),'worker':self.build_state(state_container, self.worker_state_list)}
        self.num_moves = 0
        self.substep = 0
        self.manager_timesteps = 1
        # self.count_test = 0
        if self.cirriculum and self.count >= 10000:
            self.tholds['CONTACT_SCALING'] = 0
            self.cirriculum = False

    def set_reset_point(self,point):
        '''
        Function to set a reset start point for all subsequent resets
        Intended to speed up evaluation of trained policy by allowing
        multiprocessing'''
        # print('SETTING RESET POINT', point)
        self.eval_point = point

    def step(self, action, viz=False,hand_type=None,direct_control=None):
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
        # print(agent, action)
        self._cumulative_rewards[agent] = 0

        if self._agent_selector.is_last():
            self.manipulation_phase.gym_pre_step(action)
            self.manipulation_phase.execute_action(viz=viz)
            done = self.manipulation_phase.exit_condition()
            self.manipulation_phase.post_step()
            if direct_control is not None:
                # print('ASSUMING DIRECT CONTROL')
                self.env.set_obj_pose(direct_control)
            state_container, reward_container = self.manipulation_phase.get_episode_info()
            # At this point we have the updated state after taking a move
            # So now we need to fill the observations for the next timestep?
            # Thinking we definitely fill the managers observation
            # We can add the other one elsewhere
            # rewards for all agents are placed in the .rewards dictionary
            # Now that we have actually moved we can get rewards
            self.rewards['manager'],_ = self.build_manager_reward(reward_container, self.tholds)
            # This one might not work well with the time difference. maybe need to set it up to only build rewards
            # on the last step for the manager
            self.rewards['worker'],_ = self.build_worker_reward(reward_container, self.tholds)

            # The truncations dictionary must be updated for all players.
            self.truncations = {
                agent:done for agent in self.agents
            }
            # manager gets new state
            self.state['manager'] = state_container
            self.observations['manager'] = self.build_state(state_container, self.manager_state_list)
            if self.eval or self.small_enough:
                self.record.record_timestep()
            self.timestep += 1
            if self.substep >= self.manager_timesteps - 1:
                self.agent_selection = self._agent_selector.next()
            self.substep += 1
            # if done:
            #     print('done in worker step')

        else:
            normalized_action = (action+1) * self.manager_normalizer['diff']/2 + self.manager_normalizer['mins']
            normalized_action=normalized_action.tolist()
            # print('goal to manip phase', normalized_action)
            self.manipulation_phase.set_goal(normalized_action)
            state_container = self.manipulation_phase.get_state()
            done=False
            # done = self.manipulation_phase.exit_condition()
            # if done:
            #     print('done in manager step')
            #     print(state_container['goal_pose'])
            self.state['worker'] = state_container
            state = self.build_state(state_container, self.worker_state_list)
            # print(state)
            # necessary so that observe() returns a reasonable observation at all times.
            # no rewards are allocated until both players give an action
            self._clear_rewards()
            self.observations['worker'] = state
            self.timestep += 1
            self.agent_selection = self._agent_selector.next()
            self.substep = 0

        if done:
            if self.eval or self.small_enough:
                self.record.record_episode(self.episode_type)
                if self.eval:
                    self.record.save_episode(self.episode_type, hand_type=self.hand_type)
                else:
                    self.record.save_episode(self.episode_type)

        # print(self.observations)
        self._accumulate_rewards()
        # print(self.timestep)
    
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
        # print('building state', state_container['goal_pose'].keys())
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
                    elif key == 'tstep':
                        state.append(state_container['previous_state'][i]['goal_pose']['timesteps_remaining'])
                    elif key == 'gp':
                        state.extend(state_container['previous_state'][i]['goal_pose']['upper_goal_position'])
                    elif key == 'ga':
                        state.extend(state_container['previous_state'][i]['goal_pose']['goals_open'])
                    elif key == 'go':
                        state.append(state_container['previous_state'][i]['goal_pose']['upper_goal_orientation'])
                    elif key == 'gf':
                        state.extend(state_container['previous_state'][i]['goal_pose']['goal_finger'])
                    elif key == 'mims':
                        pass
                    elif key =='lgp':
                        state.extend(state_container['previous_state'][i]['goal_pose']['goal_position'])
                    elif key == 'lgo':
                        state.append(state_container['previous_state'][i]['goal_pose']['goal_orientation'])
                    elif key == 'lgfs':
                        state.extend(state_container['previous_state'][i]['goal_pose']['goal_finger'])
                    else:
                        raise Exception(f'key {key} does not match list of known keys')

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
            elif key == 'tstep':
                state.append(state_container['goal_pose']['timesteps_remaining'])
            elif key == 'gp':
                state.extend(state_container['goal_pose']['upper_goal_position'])
            elif key == 'ga':
                state.extend(state_container['goal_pose']['goals_open'])
            elif key == 'go':
                state.append(state_container['goal_pose']['upper_goal_orientation'])
            elif key == 'gf':
                state.extend(state_container['goal_pose']['goal_finger'])
            elif key == 'mims':
                state = state_container['image']
            elif key =='lgp':
                state.extend(state_container['goal_pose']['goal_position'])
            elif key == 'lgo':
                state.append(state_container['goal_pose']['goal_orientation'])
            elif key == 'lgfs':
                state.extend(state_container['goal_pose']['goal_finger'])
            else:
                raise Exception(f'key {key} does not match list of known keys')
        return np.array(state)

    def evaluate(self, ht=None):
        print('EVALUATE TRIGGERED')
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
        # t1 = time.time()
        # print('Train triggered!!!!!!')
        self.eval = False
        self.manipulation_phase.eval = False
        self.manipulation_phase.state.train()
        self.manipulation_phase.state.reset()
        self.reset()
        self.episode_type = 'train'
        # print(time.time()-t1)