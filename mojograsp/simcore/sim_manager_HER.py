#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 11:29:05 2023

@author: orochi
"""
# pybullet imports
from mojograsp.simcore.state import StateDefault
import pybullet as p
# mojograsp module imports
from mojograsp.simcore.environment import Environment, EnvironmentDefault
from mojograsp.simcore.phase import Phase
from mojograsp.simcore.episode import Episode, EpisodeDefault
from mojograsp.simcore.record_data import RecordData, RecordDataDefault
from mojograsp.simcore.replay_buffer import ReplayBuffer, ReplayBufferDefault
from mojograsp.simcore.phase_manager import PhaseManager
from mojograsp.simcore.state import State, StateDefault
from mojograsp.simcore.reward import Reward, RewardDefault
from mojograsp.simcore.action import Action, ActionDefault
from mojograsp.simcore.sim_manager import SimManager

# python imports
import time
import os
from abc import ABC, abstractmethod
import logging
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np

def calc_finger_poses(angles):
    x0 = [-0.02675, 0.02675]
    y0 = [0.053, 0.053]
    # print('angles', angles)
    f1x = x0[0] - np.sin(angles[0])*0.072 - np.sin(angles[0] + angles[1])*0.072
    f2x = x0[1] - np.sin(angles[2])*0.072 - np.sin(angles[2] + angles[3])*0.072
    f1y = y0[0] + np.cos(angles[0])*0.072 + np.cos(angles[0] + angles[1])*0.072
    f2y = y0[1] + np.cos(angles[2])*0.072 + np.cos(angles[2] + angles[3])*0.072
    return [f1x, f1y, f2x, f2y]
    

class SimManagerRLHER(SimManager):
    """
    SimManagerRL class, runs through all episodes and uses the passed in
    mojograsp classes to call the necessary functions. This default will work on most RL applications.  
    """

    def __init__(self, num_episodes: int = 1, env: Environment = EnvironmentDefault(),
                 episode: Episode = EpisodeDefault(), record_data: RecordData = RecordDataDefault(),
                 replay_buffer: ReplayBuffer = ReplayBufferDefault, state: State = StateDefault,
                 action: Action = ActionDefault, reward: Reward = RewardDefault, args=None):
        """
        Constructor passes in the environment, episode and record data objects and simmanager parameters.

        :param num_episodes: Number of episodes to run.
        :param env: :func:`~mojograsp.simcore.environment.Environment` object
        :param episode: :func:`~mojograsp.simcore.episode.Episode` object
        :param record_data: :func:`~mojograsp.simcore.record_data.RecordData` object
        :param state: :func:`~mojograsp.simcore.state.State` object
        :param action: :func:`~mojograsp.simcore.action.Action` object
        :param reward: :func:`~mojograsp.simcore.reward.Reward` object
        :param replay_buffer: :func:`~mojograsp.simcore.replay_buffer.ReplayBuffer` object
        :type num_episodes: int
        :type env: :func:`~mojograsp.simcore.environment.Environment`
        :type episode: :func:`~mojograsp.simcore.episode.Episode`
        :type record: :func:`~mojograsp.simcore.record_data.RecordData`
        :type state: :func:`~mojograsp.simcore.state.State`
        :type action: :func:`~mojograsp.simcore.action.Action`
        :type reward: :func:`~mojograsp.simcore.reward.Reward`
        :type replay_buffer: :func:`~mojograsp.simcore.replay_buffer.ReplayBuffer`
        """
        self.state = state
        self.action = action
        self.reward = reward
        self.num_episodes = num_episodes
        self.env = env
        self.episode = episode
        self.record = record_data
        self.replay_buffer = replay_buffer
        self.phase_manager = PhaseManager()
        self.episode_number = 0
        self.record_video = False
        if args['model'] == 'DDPG+HER':
            self.use_HER = True
        else:
            self.use_HER = False
        print('USING HER:', self.use_HER)
        self.writer = SummaryWriter(args['tname'])
            
    def add_phase(self, phase_name: str, phase: Phase, start: bool = False):
        """
        Method takes in a phase name, phase object and a boolean to specify whether it is the starting phase.
        Adds the phase to the phase manager

        :param phase_name: Name of the phase being added.
        :param phase: :func:`~mojograsp.simcore.phase.Phase` object.
        :param start: If True then the phase given will be the starting phase.
        :type phase_name: str
        :type phase: :func:`~mojograsp.simcore.phase.Phase` 
        :type start: bool
        """

        self.phase_manager.add_phase(phase_name, phase, start)

    def run(self, test_flag = False):
        """
        Runs through all episodes, and the phases in each episode, continues until all episodes are completed.
        Calls the user defined phases and other classes inherited from the abstract base classes. Detailed diagram
        is above of the order of operations.
        """
        
        
        if len(self.phase_manager.phase_dict) > 0:
            logging.info("RUNNING PHASES: {}".format(
                self.phase_manager.phase_dict))
            S = None
            A = None
            R = None
            S2 = None
            E = self.episode_number

            for i in range(self.num_episodes):
                self.episode_number += 1
                self.episode.setup()
                self.env.setup()
                self.phase_manager.set_exit_flag(False)
                self.phase_manager.setup()
                print('Episode ',self.episode_number,' goal pose', self.phase_manager.current_phase.goal_position)
                print('Epsilon ', self.phase_manager.current_phase.controller.epsilon)
                timestep_number = 0
                transition_list = []
                diff_max = 0
                while self.phase_manager.get_exit_flag() == False:
                    
                    self.phase_manager.current_phase.train()
                    self.phase_manager.current_phase.setup()
                    done = False
                    logging.info("CURRENT PHASE: {}".format(
                        self.phase_manager.current_phase.name))
                    while not done:
                        # print('new timestep')
                        timestep_number += 1
                        self.phase_manager.current_phase.pre_step()
                        # self.state.set_state()
                        S = self.state.get_state()
                        if timestep_number > 1 and test_flag:
                            assert S==S2, 'This state and previous next state dont match'
                            angles = [i for i in S['two_finger_gripper']['joint_angles'].values()]
                            finger_pos = calc_finger_poses(angles)
                            S_finger_pos = [S['f1_pos'][0], S['f1_pos'][1], S['f2_pos'][0], S['f2_pos'][1]]
                            # assert np.isclose(finger_pos, S_finger_pos, atol=0.001).all(), 'calculated finger pose and pybullet finger pose dont match'
                        self.phase_manager.current_phase.execute_action()
                        A = self.action.get_action()
                        if test_flag:
                            acts = np.array(A['actor_output']) * 0.001
                            assert (np.abs(A['actor_output'])<=1).all(), 'The actor output isnt between -1 and 1'
                        self.env.step()
                        
                        done = self.phase_manager.current_phase.exit_condition()
                        
                        self.phase_manager.current_phase.post_step()
                        
                        self.record.record_timestep()
                        R = self.reward.get_reward()
                        self.state.set_state()
                        S2 = self.state.get_state()

                        if test_flag:
                            S_finger_pos = np.array([S['f1_pos'][0], S['f1_pos'][1], S['f2_pos'][0], S['f2_pos'][1]])
                            S2_finger_pos = np.array([S2['f1_pos'][0], S2['f1_pos'][1], S2['f2_pos'][0], S2['f2_pos'][1]])
                            finger_pos_change = np.linalg.norm(S_finger_pos - S2_finger_pos)
                            diff_max = max(diff_max,finger_pos_change)
                            obj_pos = S2['obj_2']['pose'][0][0:2]
                            goal_diff = [obj_pos[0]-S2['goal_pose']['goal_pose'][0],obj_pos[1]-(S2['goal_pose']['goal_pose'][1]+0.16)]
                            assert R['distance_to_goal'] == np.linalg.norm(goal_diff), 'goal dists not aligned'
                        E = self.episode_number
                        # transition = {'state':S, 'action':A, 'reward':R, 'next_state':S2, 'episode':E}
                        transition = (S,A,R,S2,E,0,timestep_number)
                        self.replay_buffer.add_timestep(transition)
                        transition_list.append(transition)
                        # done = self.phase_manager.current_phase.exit_condition()
                        if done:
                            print('done')
                        self.phase_manager.current_phase.controller.train_policy()
                        if self.record_video:
                            img = p.getCameraImage(640, 480, renderer=p.ER_BULLET_HARDWARE_OPENGL)
                            img = Image.fromarray(img[2])
                            img.save('/home/orochi/mojo/mojo-grasp/demos/rl_demo/data/vizualization/episode_' + str(self.episode_number) + '_frame_'+ str(timestep_number)+'.png')
                    self.phase_manager.get_next_phase()
                    if self.use_HER:
                        self.add_hindsight(transition_list)
                self.record.record_episode()
                self.record.save_episode()
                self.episode.post_episode()
                self.env.reset()
                # print('BEST DIFF MAX', diff_max)
                # self.writer.add_scalar('rewards/average reward', self.replay_buffer.get_average_reward(
                #     400), self.episode_number/self.num_episodes)
                # self.writer.add_scalar('rewards/min reward', self.replay_buffer.get_min_reward(400),
                #                         self.episode_number / self.num_episodes)
                # self.writer.add_scalar('rewards/max reward', self.replay_buffer.get_max_reward(400),
                #                         self.episode_number / self.num_episodes)
            # self.record.save_all()
            # self.replay_buffer.save_buffer('./data/temp_buffer.pkl')
        else:
            logging.warn("No Phases have been added")
        # print("COMPLETED")

    def add_hindsight(self,transitions):
        goal_position = transitions[-1][0]['obj_2']['pose'][0][0:2]
        end_goal = goal_position.copy()
        end_goal[1] = end_goal[1] - 0.16 
        for transition in transitions:
            transition[0]['goal_pose']['goal_pose'] = end_goal
            transition[2]['goal_position'] = goal_position
            transition[2]['distance_to_goal'] = np.sqrt((goal_position[0]-transition[0]['obj_2']['pose'][0][0])**2+(goal_position[1]-transition[0]['obj_2']['pose'][0][1])**2)
            self.replay_buffer.add_timestep(transition)
        # 
    def stall(self):
        super().stall()

    def evaluate(self):
        """
        Runs through all episodes, and the phases in each episode, continues until all episodes are completed, always using the policy
        Calls the user defined phases and other classes inherited from the abstract base classes. Detailed diagram
        is above of the order of operations. Episodes are not added to the replay buffer and agent is not trained
        """
        if len(self.phase_manager.phase_dict) > 0:
            logging.info("RUNNING PHASES: {}".format(
                self.phase_manager.phase_dict))
            end_dists = []
            
            for i in range(self.num_episodes):
                print('recording to episode', i)
                # self.episode_number += 1
                self.episode.setup()
                self.env.setup()
                self.phase_manager.set_exit_flag(False)
                self.phase_manager.setup()

                timestep_number = 0
                while self.phase_manager.get_exit_flag() == False:
                    self.phase_manager.current_phase.evaluate()
                    self.phase_manager.current_phase.setup()
                    
                    done = False
                    logging.info("CURRENT PHASE: {}".format(
                        self.phase_manager.current_phase.name))
                    while not done:
                        timestep_number += 1
                        self.phase_manager.current_phase.pre_step()
                        # self.state.set_state()
                        # S = self.state.get_state()
                        self.phase_manager.current_phase.execute_action()
                        # A = self.action.get_action()
                        self.env.step()
                        done = self.phase_manager.current_phase.exit_condition(True)
                        self.phase_manager.current_phase.post_step()
                        self.record.record_timestep()
                        R = self.reward.get_reward()
                        self.state.set_state()
                        # S2 = self.state.get_state()
                        # E = self.episode_number
                        # transition = (S, A, R, S2, E)
                        # self.replay_buffer.add_timestep(transition)
                        # done = self.phase_manager.current_phase.exit_condition()
                        # self.phase_manager.current_phase.controller.train_policy()
                        # print(self.record_video)
                        if self.record_video:
                            
                            img = p.getCameraImage(640, 480, renderer=p.ER_BULLET_HARDWARE_OPENGL)
                            img = Image.fromarray(img[2])
                            img.save('/home/orochi/mojo/mojo-grasp/demos/rl_demo/data/vizualization/Evaluation_episode_' + str(i) + '_frame_'+ str(timestep_number)+'.png')
                    self.phase_manager.get_next_phase()
                    end_dists.append(R['distance_to_goal'])
                self.record.record_episode(True)
                self.record.save_episode(True)
                self.episode.post_episode()
                self.env.reset()
                # self.writer.add_scalar('rewards/average reward', self.replay_buffer.get_average_reward(
                #     400), self.episode_number/self.num_episodes)
                # self.writer.add_scalar('rewards/min reward', self.replay_buffer.get_min_reward(400),
                #                         self.episode_number / self.num_episodes)
                # self.writer.add_scalar('rewards/max reward', self.replay_buffer.get_max_reward(400),
                #                         self.episode_number / self.num_episodes)
            # self.record.save_all()
            # self.replay_buffer.save_buffer('./data/temp_buffer.pkl')
            avg_dist = np.mean(end_dists)
            avg_dist_err = np.std(end_dists)
            print(end_dists)
            print(f'evaluated, average ending distance was {avg_dist} +/- {avg_dist_err}')
        else:
            logging.warn("No Phases have been added")
    
    def save_network(self, filename):
        self.phase_manager.current_phase.controller.policy.save(filename)
    
    def replay(self, action_list):
        """
        Runs through a single episode following the actions in the action list
        Calls the user defined phases and other classes inherited from the abstract base classes. Detailed diagram
        is above of the order of operations. Episodes are not added to the replay buffer and agent is not trained
        """
        if len(self.phase_manager.phase_dict) > 0:
            logging.info("RUNNING PHASES: {}".format(
                self.phase_manager.phase_dict))

            for i in range(self.num_episodes):
                # self.episode_number += 1
                self.episode.setup()
                self.env.setup()
                self.phase_manager.set_exit_flag(False)
                self.phase_manager.setup()

                timestep_number = 0
                while self.phase_manager.get_exit_flag() == False:
                    self.phase_manager.current_phase.setup()
                    done = False
                    logging.info("CURRENT PHASE: {}".format(
                        self.phase_manager.current_phase.name))
                    while not done:
                        
                        self.phase_manager.current_phase.pre_step()
                        self.state.set_state()
                        # S = self.state.get_state()
                        self.phase_manager.current_phase.execute_action(action_list[timestep_number])
                        timestep_number += 1
                        # A = self.action.get_action()
                        self.env.step()
                        done = self.phase_manager.current_phase.exit_condition()
                        self.phase_manager.current_phase.post_step()
                        self.record.record_timestep()
                        # R = self.reward.get_reward()
                        self.state.set_state()
                        # S2 = self.state.get_state()
                        # E = self.episode_number
                        # transition = (S, A, R, S2, E)
                        # self.replay_buffer.add_timestep(transition)
                        # done = self.phase_manager.current_phase.exit_condition()
                        # self.phase_manager.current_phase.controller.train_policy()
                        if self.record_video:
                            img = p.getCameraImage(640, 480, renderer=p.ER_BULLET_HARDWARE_OPENGL)
                            img = Image.fromarray(img[2])
                            img.save('/home/orochi/mojo/mojo-grasp/demos/rl_demo/data/vizualization/Evaluation_episode_' + str(self.episode_number) + '_frame_'+ str(timestep_number)+'.png')
                    self.phase_manager.get_next_phase()
                self.record.record_episode(True)
                self.record.save_episode(True)
                self.episode.post_episode()
                self.env.reset()
                # self.writer.add_scalar('rewards/average reward', self.replay_buffer.get_average_reward(
                #     400), self.episode_number/self.num_episodes)
                # self.writer.add_scalar('rewards/min reward', self.replay_buffer.get_min_reward(400),
                #                         self.episode_number / self.num_episodes)
                # self.writer.add_scalar('rewards/max reward', self.replay_buffer.get_max_reward(400),
                #                         self.episode_number / self.num_episodes)
            # self.record.save_all()
            # self.replay_buffer.save_buffer('./data/temp_buffer.pkl')
        else:
            logging.warn("No Phases have been added")