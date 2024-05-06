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
    Example environment that follows gym interface to allow us to use openai gym learning algorithms with mojograsp
    '''
    
    def __init__(self, rl_env, manipulation_phase,record_data, args):
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
        self.ROTATION_SCALING = args['rotation_scaling']
        self.image_path = args['save_path'] + 'Videos/'
        self.record = record_data
        self.eval = False
        self.viz = False
        self.eval_run = 0
        self.timestep = 0
        self.past_time = time.time()
        self.thing = []
        self.eval_point = None
        self.hand_type = None
        self.first = True
        self.small_enough = args['epochs'] <= 100000
        self.OBJECT_POSE_RANDOMIZATION = args['object_random_start']
        try:
            self.DOMAIN_RANDOMIZATION_MASS = args['domain_randomization_object_mass']
            self.DOMAIN_RANDOMIZATION_FINGER = args['domain_randomization_finger_friction']
            self.DOMAIN_RANDOMIZATION_FLOOR = args['domain_randomization_floor_friction']
        except KeyError:
            self.DOMAIN_RANDOMIZATION_MASS = False
            self.DOMAIN_RANDOMIZATION_FINGER = False
            self.DOMAIN_RANDOMIZATION_FLOOR = False
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
                       'ROTATION_SCALING':self.ROTATION_SCALING,
                       'SUCCESS_REWARD':self.SUCCESS_REWARD}

    def prep_reward(self):
        """
        Method takes in a Reward object
        Extracts reward information from state_container and returns it as a float
        based on the reward structure contained in self.REWARD_TYPE

        :param state: :func:`~mojograsp.simcore.reward.Reward` object.
        :type state: :func:`~mojograsp.simcore.reward.Reward`
        """        
        # print(self.TASK,self.REWARD_TYPE)
        if 'Rotation' in self.TASK:
            if self.TASK == 'Rotation+Finger':
                print('rotation and finger')
                self.build_reward = rf.rotation_with_finger
            elif (self.TASK == 'Rotation_single')|(self.TASK =='Rotation_region'):
                print('just rotation no sliding, stay in place dammit, added finger. make sure contact scaling is 0 if no finger desired')
                self.build_reward = rf.rotation_with_finger
        elif 'contact' in self.TASK:
            self.build_reward = rf.contact_point 
        elif self.TASK =='full_task':
            print('All them rotation and sliding')
            self.build_reward = rf.slide_and_rotate 
        else:
            if self.REWARD_TYPE == 'Sparse':
                self.build_reward = rf.sparse
            elif self.REWARD_TYPE == 'Distance':
                self.build_reward = rf.distance
            elif self.REWARD_TYPE == 'Distance + Finger':
                self.build_reward = rf.distance_finger
            elif self.REWARD_TYPE == 'Hinge Distance + Finger':
                self.build_reward = rf.hinge_distance
            elif self.REWARD_TYPE == 'Slope':
                self.build_reward = rf.slope
            elif self.REWARD_TYPE == 'Slope + Finger':
                self.build_reward = rf.slope_finger
            elif self.REWARD_TYPE == 'SmartDistance + Finger':
                self.build_reward = rf.smart
            elif self.REWARD_TYPE == 'ScaledDistance + Finger':
                self.build_reward = rf.scaled
            elif (self.REWARD_TYPE == 'ScaledDistance+ScaledFinger') and (self.TASK != 'multi'):
                self.build_reward = rf.double_scaled
            elif 'wall' in self.TASK:
                self.build_reward = rf.double_scaled
            elif self.REWARD_TYPE == 'TripleScaled':
                self.build_reward = rf.triple_scaled_slide
            elif self.REWARD_TYPE == 'SFS':
                self.build_reward = rf.sfs
            elif self.REWARD_TYPE == 'DFS':
                self.build_reward = rf.dfs
            elif self.REWARD_TYPE == 'SmartDistance + SmartFinger':
                self.build_reward = rf.double_smart
            elif (self.TASK == 'multi') and (self.REWARD_TYPE =='ScaledDistance+ScaledFinger'):
                self.build_reward = rf.multi_scaled
            else:
                raise Exception('reward type does not match list of known reward types')

    def seed(self,seed):
        '''
        set individual random seed
        currently unused
        '''
        self._seed = seed

    def set_reset_point(self,point):
        '''
        Function to set a reset start point for all subsequent resets
        Intended to speed up evaluation of trained policy by allowing
        multiprocessing'''
        print('SETTING RESET POINT', point)
        self.eval_point = point

    def reset(self,special=None):

        if not self.first:
            self.thing.append(time.time()-self.past_time)
            self.past_time = time.time()
            if self.manipulation_phase.episode >= self.manipulation_phase.state.objects[-1].len:
                self.manipulation_phase.reset()
                # print('average time of episode',np.average(self.thing))
                self.thing = []
            new_goal,fingerys = self.manipulation_phase.next_ep()
            # print('new goal from reset', new_goal)
        else:
            new_goal = {'goal_position':[0,0]}
            fingerys = [0,0]

        self.timestep=0
        self.first = False
        self.env.apply_domain_randomization(self.DOMAIN_RANDOMIZATION_FINGER,self.DOMAIN_RANDOMIZATION_FLOOR,self.DOMAIN_RANDOMIZATION_MASS)
        
        if self.eval:
            self.eval_run +=1

        if self.eval_point is not None:
            # print('eval point and goal ', self.eval_point, new_goal)
            self.env.reset(self.eval_point)
        elif type(special) is list:
            self.env.reset_to_pos(special[0],special[1])
        elif type(special) is dict:
            print('reseting with special dict', special)
            if 'fingers' in special.keys():
                self.env.reset(special['goal_position'], special['fingers'])
            else:
                self.env.reset(special['goal_position'])
        elif (self.TASK == 'Rotation_region') | ('contact' in self.TASK):
            self.env.reset(new_goal['goal_position'],fingerys=fingerys)
        elif self.OBJECT_POSE_RANDOMIZATION:
            random_start = np.random.uniform(0,1,2)
            x = (1-random_start[0]**2) * np.sin(random_start[1]*2*np.pi) * 0.06
            y = (1-random_start[0]**2) * np.cos(random_start[1]*2*np.pi) * 0.04
            # print('x and y',x,y)
            self.env.reset([x,y])
        elif 'wall' in self.TASK:
            self.env.reset([0.0463644396618753, 0.012423314164921])
        else:
            # print('reseting with NO parameters')
            self.env.reset()
        self.manipulation_phase.setup()
        
        state, _ = self.manipulation_phase.get_episode_info()
        if state['goal_pose']['goal_finger'] is not None:
            self.env.set_finger_contact_goal(state['goal_pose']['goal_finger'])

        state = self.build_state(state)
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
        if self.discrete:
            action = action-1
            # print(action)
        # print('going to manipulation_phase')
        self.manipulation_phase.gym_pre_step(action)
        # print('executing action')
        self.manipulation_phase.execute_action(viz=viz)
        done = self.manipulation_phase.exit_condition(self.eval)
        self.manipulation_phase.post_step()
        
        if self.eval or self.small_enough:
            self.record.record_timestep()
        # print('recorded timesteps')
        state, reward_container = self.manipulation_phase.get_episode_info()

        info = {}
        if mirror:
            state = self.build_mirror_state(state)
        else:
            state = self.build_state(state)
        if self.STATE_NOISE > 0:
            state = self.noisey_boi.add_noise(state, self.STATE_NOISE)
        reward, done2 = self.build_reward(reward_container, self.tholds)

        if self.TASK == 'multi':
            if done2 and not done:
                # print('doing next goal')
                # print(reward)
                self.manipulation_phase.next_goal()
        else:
            done = done | done2

        
        if done:
            # print('done, recording stuff')
            if self.eval or self.small_enough:
                self.record.record_episode(self.episode_type)
                if self.eval:
                    if self.hand_type is None:
                        self.record.save_episode(self.episode_type, hand_type=hand_type)
                    else:
                        self.record.save_episode(self.episode_type, hand_type=self.hand_type)
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
                    elif key == 'wall':
                        state.extend(state_container['previous_state'][i]['wall']['pose'][0][0:2])
                        state.extend(state_container['previous_state'][i]['wall']['pose'][1][0:4])
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
                # print(state)
            elif key == 'go':
                state.append(state_container['goal_pose']['goal_orientation'])
            elif key == 'gf':
                state.extend(state_container['goal_pose']['goal_finger'])
            elif key == 'wall':
                state.extend(state_container['wall']['pose'][0][0:2])
                state.extend(state_container['wall']['pose'][1][0:4])
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
        self.hand_type = None

    def set_goal(self,goal):
        self.env.set_goal(goal)