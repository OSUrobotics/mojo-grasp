#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 10:33:32 2022

@author: orochi
"""

from mojograsp.simcore.state import StateDefault
import numpy as np
from copy import deepcopy
from mojograsp.simobjects.two_finger_gripper import TwoFingerGripper
from mojograsp.simcore.goal_holder import *
# from point_generator import slice_obj_at_y_level, calculate_outer_perimeter, find_intersection_points
from mojograsp.simcore.image_maker import ImageGenerator
import demos.rl_demo.point_generator as pg
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from demos.rl_demo.autoencoder import Autoencoder, load_trained_model
import pickle as pkl
from scipy.spatial.transform import Rotation as R
import os

class DictHolder():
    def __init__(self,list_size):
        self.data = []
        self.max_len = list_size

    def reset(self):
        self.data = []

    def append(self, data: dict):
        self.data.append(data)
        if len(self.data) > self.max_len:
            self.data.pop(0)

    def get_full(self):
        data_dict = self.data[-1].copy()
        if len(self.data) < self.max_len:
            data_dict['previous_state'] = [self.data[0] if i<self.max_len-len(self.data) else self.data[i] for i in range(self.max_len-1)]
        else:
            data_dict['previous_state'] = self.data[0:-1]
        return data_dict

class MultiprocessState(StateDefault):
    """
    Default State Class that is used when the user does not need or wish to use the Action class
    but it may still be called by other classes such as :func:`~mojograsp.simcore.record_data.RecordData` 
    and :func:`~mojograsp.simcore.replay_buffer.ReplayBufferDefault`.
    """

    def __init__(self, pybullet_instance, objects: list = None, prev_len=0, physicsClientId=None,eval_goals:GoalHolder =None):
        """
        Default placeholder constructor optionally takes in a list of objects, if no list is provided it defaults
        to None. 

        :param objects: list of mojograsp objects.
        :type objects: list
        """
        super().__init__()
        self.p = pybullet_instance
        dirname, filename = os.path.split(os.path.abspath(__file__))
        # print(dirname)
        self.encoder = load_trained_model(dirname+'/test_best_autoencoder_16.pth',72,16,72)
        self.encoder.eval()
        with open(dirname+"/test_input_scaler.pkl", "rb") as f:
            self.loaded_scaler = pkl.load(f)
        with open(dirname+"/test_output_scaler.pkl", "rb") as f:
            self.output_scaler = pkl.load(f)
        self.objects = objects 
        obj_path = self.objects[1].get_path()
        self.ori_corrector = [0,0,0,0]
        #print('OBJ PATH', obj_path)
        self.slice = pg.get_slice(obj_path)
        self.rotated_static = self.slice
        #print(len(self.slice))
        for object in self.objects:
            if type(object) == TwoFingerGripper:
                temp = object.link_lengths
                self.hand_params = [temp[0][0][1],temp[0][1][1],temp[1][0][1],temp[1][1][1], object.palm_width]
                self.hand_name = object.record_name
        if prev_len > 0:            
            self.previous_states = [{}]*prev_len
            # self.state_holder = DictHolder(prev_len)
            self.pflag = True
        else:
            self.pflag = False
        if eval_goals is not None:
            self.eval_goals = eval_goals
            self.train_goals = deepcopy(self.objects[-1])
            self.train_flag = True
        else:
            self.eval_goals = None
        self.image_gen = ImageGenerator((240,240,1))
            
    def evaluate(self):
        if (self.eval_goals is not None) and self.train_flag:
            self.train_flag = False
            self.objects[-1] = self.eval_goals
            # print('did an evaluate', self.eval_goals.pose[1],self.train_goals.pose[1])
            
    def train(self):
        if (self.eval_goals is not None) and not self.train_flag:
            self.train_flag = True
            self.objects[-1] = self.train_goals
            # print('did a train', self.train_goals.pose[1])
            
    def next_run(self):
        for thing in self.objects:
            if (type(thing) == GoalHolder) | (type(thing) == RandomGoalHolder) | (type(thing) == SingleGoalHolder)|(type(thing) == HRLGoalHolder)|(type(thing) == HRLMultigoalHolder) |(type(thing) == HRLMultigoalFixed)| (type(thing) == HRLMultigoalFixedPaired):
                fingerys = thing.next_run()
                temp = thing.get_data()
        return temp, fingerys
    
    def get_run_start(self):
        for thing in self.objects:
            if (type(thing) == GoalHolder) | (type(thing) == RandomGoalHolder) | (type(thing) == SingleGoalHolder)|(type(thing) == HRLGoalHolder)|(type(thing) == HRLMultigoalHolder) |(type(thing) == HRLMultigoalFixed)| (type(thing) == HRLMultigoalFixedPaired):
                fingerys = thing.get_finger_start()
                temp = thing.get_data()
        return temp, fingerys
    
    def reset(self):
        self.run_num = 0
        for thing in self.objects:
            if (type(thing) == GoalHolder) | (type(thing) == RandomGoalHolder):
                thing.reset()
    
    def get_hand_name(self):
        # print('got hand name',self.hand_name)
        return self.hand_name
    
    def calc_distance(self,p1,p2):
        return np.sqrt((p2[0]-p1[0])**2+(p2[1]-p1[1])**2)
    
    def wrap_angle(self, angle):
        angle = angle % (2*np.pi)
        angle = (angle + 2*np.pi) % (2*np.pi)
        if angle > np.pi:
            angle = angle - 2*np.pi
        return angle

    def calc_contact_angle(self):
        obj_angle = self.current_state['obj_2']['pose'][1][2] + np.pi/2 #Get yaw
        #print('START F1 CONTACT POS \n')
        #print(self.current_state['f1_contact_pos'][0])
        #print(self.current_state['obj_2']['pose'][0])
        #print('\n')
        #print('END F1 CONTACT POS \n')
        f1_contact_vector_angle = np.arctan2((self.current_state['f1_contact_pos'][1] - self.current_state['obj_2']['pose'][0][1]),(self.current_state['f1_contact_pos'][0]- self.current_state['obj_2']['pose'][0][0])) #Get the vector angle from the contact point to the object
        f2_contact_vector_angle = np.arctan2((self.current_state['f2_contact_pos'][1] - self.current_state['obj_2']['pose'][0][1]),(self.current_state['f2_contact_pos'][0]- self.current_state['obj_2']['pose'][0][0])) #Get the vector angle from the contact point to the object
        f1_contact_angle = self.wrap_angle(f1_contact_vector_angle - obj_angle)
        f2_contact_angle = self.wrap_angle(f2_contact_vector_angle - obj_angle)
        return f1_contact_angle, f2_contact_angle 
    
    def check_contact(self):
        if len(self.p.getContactPoints(self.objects[1].id, self.objects[0].id, 1)) > 0 or len(self.p.getContactPoints(self.objects[1].id, self.objects[0].id, 2)) > 0:
            f1 = 1
        else : f1 = 0
        if len(self.p.getContactPoints(self.objects[1].id, self.objects[0].id, 4)) > 0 or len(self.p.getContactPoints(self.objects[1].id, self.objects[0].id, 5)) > 0:
            f2 = 1
        else : f2 = 0
        return f1,f2

    def get_dynamic(self, shape, pose, orientation):
        """
        Computes the dynamic state of the object by applying a quaternion rotation 
        and translation to the input shape.
        """
        shape = np.hstack((shape, np.full((shape.shape[0], 1), 0.0)))

        x, y, z = pose
        quaternion = np.array(orientation)

        rotation_matrix = R.from_quat(quaternion).as_matrix()

        shape = shape @ rotation_matrix.T

        shape[:, 0] += x
        shape[:, 1] += y
        shape[:, 2] += z

        return shape, rotation_matrix[:2, :].flatten().tolist()
    
    def decode_latent(self, model, latent_vector, output_scaler):
        """
        Decodes a given latent space representation and scales the output back to its original scale.
        
        Parameters:
        - model: The trained autoencoder model
        - latent_vector: A tensor or numpy array representing the latent space input
        - output_scaler: Scaler used to inverse transform the output back to original scale
        - output_dim: Number of output dimensions (static representation size)
        
        Returns:
        - A tuple containing:
            - A list of 3 elements
            - A list of 4 elements
            - A list of 24 (x, y) coordinate pairs
        """
        # Ensure the input is a PyTorch tensor
        if not isinstance(latent_vector, torch.Tensor):
            latent_vector = torch.tensor(latent_vector, dtype=torch.float32)
        
        # Ensure correct shape (batch size of 1)
        if latent_vector.dim() == 1:
            latent_vector = latent_vector.unsqueeze(0)  # Shape: (1, latent_dim)
        
        # Decode the latent vector
        with torch.no_grad():
            reconstruction = model.decode(latent_vector)  # shape: (1, output_dim)
        
        # Convert to numpy and inverse transform
        reconstruction_np = reconstruction.cpu().numpy()
        reconstruction_unscaled = output_scaler.inverse_transform(reconstruction_np)[0]
        
        # Reshape the output into the required tuple format
        part_1 = reconstruction_unscaled[:3].tolist()  # 3 elements
        part_2 = reconstruction_unscaled[3:7].tolist()  # 4 elements
        part_3 = reconstruction_unscaled[7:].reshape(-1, 2).tolist()  # 24 (x, y) pairs
        
        return (part_1, part_2, part_3)
    
    def correct_ori(self, corrector, current_orientation):
        """
        Corrects the orientation of the object based on the given corrector and current orientation.
        
        Parameters:
        - corrector: The corrector quaternion
        - current_orientation: The current orientation quaternion
        
        Returns:
        - A list containing the corrected orientation
        """
        # Convert to numpy arrays
        corrector = np.array(corrector)
        current_orientation = np.array(current_orientation)
        
        # Perform quaternion multiplication
        corrected_orientation = R.from_quat(corrector) * R.from_quat(current_orientation)
        
        # Return the corrected orientation as a list
        return corrected_orientation.as_quat().tolist()


    def set_state(self):
        """
        Default method that sets self.current_state to either get_data() for the object or an empty dictionary
        """
        # print('setting state')
        if self.pflag:
            self.previous_states[1:] = self.previous_states[0:-1]
            self.previous_states[0] = deepcopy(self.current_state)
            # TODO FIGURE OUT WHY THIS DAMN THING IS FUCKED
            # SPECIFICALLY WHY THE ORIENTATION UPDATES CORRECTLY BUT THE POSITION DOES NOT
            # AND MAKE SURE THE OTHER ONES ARENT FUCKED TOO
        super().set_state()

        #temp1 = self.p.getClosestPoints(self.objects[1].id, self.objects[0].id, 10, -1, 1, -1)[0]
        #temp2 = self.p.getClosestPoints(self.objects[1].id, self.objects[0].id, 10, -1, 4, -1)[0]
        link1_pose = self.p.getLinkState(self.objects[0].id, 2)

        link2_pose = self.p.getLinkState(self.objects[0].id, 5)
        link1_base = self.p.getLinkState(self.objects[0].id, 1)

        link2_base = self.p.getLinkState(self.objects[0].id, 4)

        self.current_state['f1_pos'] = list(link1_pose[0])
        self.current_state['f2_pos'] = list(link2_pose[0])
        self.current_state['f1_base'] = list(link1_base[0])
        self.current_state['f2_base'] = list(link2_base[0])
        self.current_state['f1_ang'] = self.current_state['two_finger_gripper']['joint_angles']['finger0_segment0_joint'] + self.current_state['two_finger_gripper']['joint_angles']['finger0_segment1_joint']
        self.current_state['f2_ang'] = self.current_state['two_finger_gripper']['joint_angles']['finger1_segment0_joint'] + self.current_state['two_finger_gripper']['joint_angles']['finger1_segment1_joint']        
        #self.current_state['f1_contact_pos'] = list(temp1[6])
        #self.current_state['f2_contact_pos'] = list(temp2[6])
        self.current_state['hand_params'] = self.hand_params.copy()
        # if self.pflag:
        #     self.state_holder.append(self.current_state.copy())
        
        if 'upper_goal_position' in self.current_state['goal_pose'].keys():
            unreached_goals = [[self.current_state['goal_pose']['upper_goal_position'][2*i],self.current_state['goal_pose']['upper_goal_position'][i*2+1]] for i,v in enumerate(self.current_state['goal_pose']['goals_open']) if v]
            self.current_state['image'] = self.image_gen.draw_stamp(self.current_state['obj_2']['pose'],
                                                                unreached_goals)
        #What Jeremiah Is Adding
        self.current_state['slice'] = self.slice
        self.current_state['dynamic'], self.current_state['mat_comp'] = self.get_dynamic(self.slice,self.current_state['obj_2']['pose'][0][0:3],self.current_state['obj_2']['pose'][1])
        dynamic_np = np.array(self.current_state['dynamic'].flatten()).reshape(1, -1)
        normalized_np = self.loaded_scaler.transform(dynamic_np)
        normalized_state = torch.tensor(normalized_np, dtype=torch.float32).reshape(1, -1)
        encoder_state, _ = self.encoder(normalized_state)
        self.current_state['latent'] = encoder_state.detach().numpy()

        #ADDED April 19th
        self.current_state['corrected_orientation'] = self.correct_ori(self.ori_corrector,self.current_state['obj_2']['pose'][1])
        self.current_state['rotated_static'] = self.rotated_static
        #print(np.mean(self.current_state['latent']))
        #self.current_state['remade'] = self.decode_latent(self.encoder, self.current_state['latent'], self.output_scaler)
        
    def init_state(self):
        """
        Default method that sets self.current_state to either get_data() for the object or an empty dictionary
        """
        # print('initializing state')
        super().set_state()
        # print(self.current_state)
        #temp1 = self.p.getClosestPoints(self.objects[1].id, self.objects[0].id, 10, -1, 1, -1)[0]
        #temp2 = self.p.getClosestPoints(self.objects[1].id, self.objects[0].id, 10, -1, 4, -1)[0]
        link1_pose = self.p.getLinkState(self.objects[0].id, 2)
        link2_pose = self.p.getLinkState(self.objects[0].id, 5)
        link1_base = self.p.getLinkState(self.objects[0].id, 1)
        link2_base = self.p.getLinkState(self.objects[0].id, 4)
        self.current_state['f1_pos'] = list(link1_pose[0])
        self.current_state['f2_pos'] = list(link2_pose[0])
        self.current_state['f1_base'] = list(link1_base[0])
        self.current_state['f2_base'] = list(link2_base[0])
        self.current_state['f1_ang'] = self.current_state['two_finger_gripper']['joint_angles']['finger0_segment0_joint'] + self.current_state['two_finger_gripper']['joint_angles']['finger0_segment1_joint']
        self.current_state['f2_ang'] = self.current_state['two_finger_gripper']['joint_angles']['finger1_segment0_joint'] + self.current_state['two_finger_gripper']['joint_angles']['finger1_segment1_joint']
        #self.current_state['f1_contact_pos'] = list(temp1[6])
        #self.current_state['f2_contact_pos'] = list(temp2[6])
        self.current_state['hand_params'] = self.hand_params.copy()
        #What Jeremiah Is Adding
        if 'upper_goal_position' in self.current_state['goal_pose'].keys():
            unreached_goals = [[self.current_state['goal_pose']['upper_goal_position'][2*i],self.current_state['goal_pose']['upper_goal_position'][i*2+1]] for i,v in enumerate(self.current_state['goal_pose']['goals_open']) if v]

            self.current_state['image'] = self.image_gen.draw_stamp(self.current_state['obj_2']['pose'],
                                                                    unreached_goals)
        self.current_state['slice'] = self.slice

        self.current_state['dynamic'], self.current_state['mat_comp'] = self.get_dynamic(self.slice,self.current_state['obj_2']['pose'][0][0:3],self.current_state['obj_2']['pose'][1])

        dynamic_np = np.array(self.current_state['dynamic'].flatten()).reshape(1, -1)
        normalized_np = self.loaded_scaler.transform(dynamic_np)
        normalized_state = torch.tensor(normalized_np, dtype=torch.float32).reshape(1, -1)
        encoder_state, _ = self.encoder(normalized_state)
        self.current_state['latent'] = encoder_state.detach().numpy()
        #ADDED April 19th
        self.ori_corrector = self.current_state['obj_2']['pose'][1]
        self.current_state['corrected_orientation'] = self.correct_ori(self.ori_corrector,self.current_state['obj_2']['pose'][1])
        temp_shape = self.current_state['dynamic'].reshape(24,3)
        self.rotated_static = temp_shape[:,:2]
        self.current_state['rotated_static'] = self.rotated_static
        ##################
        #self.current_state['remade'] = self.decode_latent(self.encoder, self.current_state['latent'], self.output_scaler)

        if self.pflag:
            for i in range(len(self.previous_states)):
                self.previous_states[i] = self.current_state.copy()
        # if self.pflag:
        #     self.state_holder.append(self.current_state.copy())
         
    def get_state(self) -> dict:
        """
        Default method will return a dictionary containing the the get_data() return value for every object
        in the objects list. If no objects are given then it returns an empty dictionary.

        :return: Dictionary containing the representation of the current simulator state or an empty dictionary.
        :rtype: dict
        """
        # print('g state')
        # print('goal from get state', self.current_state['goal_pose'])

        # if self.pflag:
        #     temp = self.state_holder.get_full()
        #     print(temp)
        #     return temp
        # else:
        #     return self.current_state.copy()
        # print(self.current_state)
        temp = self.current_state.copy()
        if self.pflag:
            temp['previous_state'] = self.previous_states.copy()
        return temp
    
    def get_name(self) -> str:
        for thing in self.objects:
            if (type(thing) == GoalHolder) | (type(thing) == RandomGoalHolder):
                return thing.get_name()
    
    def set_goal(self,goal_list):
        # print('SETTING THE GOAL')
        for thing in self.objects:
            if (type(thing) == GoalHolder) | (type(thing) == RandomGoalHolder)|(type(thing) == HRLGoalHolder)|(type(thing) == HRLMultigoalHolder)|(type(thing) == HRLMultigoalFixed)| (type(thing) == HRLMultigoalFixedPaired):
                # print('setting goal', goal_list)
                if len(goal_list) > 4:
                    thing.set_pose(goal_list[0:2], goal_list[2],goal_list[3:5])
                elif len(goal_list) >=3:
                    thing.set_pose(goal_list[0:2], goal_list[2])
                else:
                    thing.set_pose(goal_list[0:2])
                self.current_state[thing.name] = thing.get_data()
                # if self.pflag:
                #     self.state_holder[-1][thing.name] = thing.get_data()
    

    def get_goal(self):
        try:
            return self.current_state['goal_pose']
        except KeyError:
            return [0,0]

    def check_goal(self):
        for thing in self.objects:
            if (type(thing) == HRLMultigoalHolder) | (type(thing) == HRLMultigoalFixed)| (type(thing) == HRLMultigoalFixedPaired):
                thing.check_goal([self.current_state['obj_2']['pose'][0][0],self.current_state['obj_2']['pose'][0][1]-0.1])

    def __eq__(self, o):
        # Doesnt check that the objects are the same or that the run number is the same,
        # only checks that the values saved in state are the same
        if isinstance(o, MultiprocessState):
            return self.current_state == o.current_state
        return False
