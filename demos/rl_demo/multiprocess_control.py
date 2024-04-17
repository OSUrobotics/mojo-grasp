from multiprocessing.dummy import current_process

import numpy as np

from mojograsp.simobjects.two_finger_gripper import TwoFingerGripper
from mojograsp.simobjects.object_base import ObjectBase
from typing import List
import numpy as np
import os
# import Markers
import torch
from mojograsp.simcore.jacobianIK.multiprocess_jik import JacobianIK
from copy import deepcopy
import time

def calc_finger_poses(angles):
    x0 = [-0.02675, 0.02675]
    y0 = [0.0, 0.0]
    print('angles', angles)
    f1x = x0[0] - np.sin(angles[0])*0.0936 - np.sin(angles[0] + angles[1])*0.0504
    f2x = x0[1] - np.sin(angles[2])*0.0936 - np.sin(angles[2] + angles[3])*0.0504
    f1y = y0[0] + np.cos(angles[0])*0.0936 + np.cos(angles[0] + angles[1])*0.0504
    f2y = y0[1] + np.cos(angles[2])*0.0936 + np.cos(angles[2] + angles[3])*0.0504
    print([f1x, f1y, f2x, f2y])
    return [f1x, f1y, f2x, f2y]

def clip_angs(angles):
    # function to clip angles to between -pi to pi

    for i,angle in enumerate(angles):
        period = np.floor((np.pi+angle)/(2*np.pi))
        if period != 0:
            # print('clipping ang1',angle)
            
            angles[i] = angles[i] - period*2*np.pi
            # print('new angle', angles[i])
    return angles

class MultiprocessController():
    # Maximum move per step
    MAX_MOVE = .01

    def __init__(self, pybullet_instance, gripper: TwoFingerGripper, cube: ObjectBase, data_file: str = None, args=None, hand_type=None):
        self.p = pybullet_instance

        self.train_flag = False
        self.MAX_ANGLE_CHANGE = 0.01
        self.MAX_DISTANCE_CHANGE = 0.001
        self.epsilon = args['epsilon']
        self.COOLING_RATE = args['edecay']
        self.rand_portion = np.array([0,0,0,0])
        self.final_reward = 0
        
        self.old_epsilon = self.epsilon
        # print('epsilon and edecay', self.epsilon, self.COOLING_RATE)
        
        self.rand_episode = np.random.rand() < self.epsilon
        self.useIK = args['action']=="Finger Tip Position"
        # self.eval_flag = False
        
        self.gripper = gripper
        self.cube = cube
        self.path = data_file
        self.end_effector_links = [1, 3]
        # print('checking velocity in controller', self.p.getBaseVelocity(cube.id))
        # world coordinates
        self.goal_position = None
        # world coordinates
        self.current_cube_pose = None
        # makes sure contact isnt gone for too long
        self.num_contact_loss = 0
        self.prev_distance = 0
        self.distance_count = 0
        self.retry_count = 0

        f1 = {"name": "finger0", "num_links": 2, "link_lengths": self.gripper.link_lengths[0]}
        f2 = {"name": "finger1", "num_links": 2, "link_lengths": self.gripper.link_lengths[1]}


        hand_info = {"finger1": f1,"finger2": f2}
        self.p.resetJointState(self.gripper.id, 0, 0)
        self.p.resetJointState(self.gripper.id, 1, 0)
        self.p.resetJointState(self.gripper.id, 3, 0)
        self.p.resetJointState(self.gripper.id, 4, 0)
        self.ik_f1 = JacobianIK(self.p, gripper.id,deepcopy(hand_info['finger1']), error=1e-4)
        self.ik_f2 = JacobianIK(self.p, gripper.id,deepcopy(hand_info['finger2']), error=1e-4)

        self.p.resetJointState(self.gripper.id, 0, self.gripper.starting_angles[0])
        self.p.resetJointState(self.gripper.id, 1, self.gripper.starting_angles[1])
        self.p.resetJointState(self.gripper.id, 3, self.gripper.starting_angles[2])
        self.p.resetJointState(self.gripper.id, 4, self.gripper.starting_angles[3])

        self.p.stepSimulation()
        self.ik_f1.finger_fk.update_angles_from_sim()
        self.ik_f2.finger_fk.update_angles_from_sim()

        if 'IK_freq' in args.keys():
            self.INTERP_IK = not args['IK_freq']
        else:
            self.INTERP_IK = False
        self.num_tsteps = int(240/args['freq'])
        self.MAX_DISTANCE_CHANGE = self.MAX_DISTANCE_CHANGE/8
        self.MAX_ANGLE_CHANGE = self.MAX_ANGLE_CHANGE/8

    def find_angles(self,actor_output):
        # print('finding angles')
        if self.useIK:
            # print('we should not see this')
            finger_pos1 = self.p.getLinkState(self.gripper.id, 2) #RIGHT FINGER
            finger_pos2 = self.p.getLinkState(self.gripper.id, 5) #LEFT FINGER
            finger_pos1 = finger_pos1[0]
            finger_pos2 = finger_pos2[0]
            finger_angles = self.gripper.get_joint_angles()
            ap = actor_output * self.MAX_DISTANCE_CHANGE
            action_list = []
            # print(f'actor_output: {actor_output}, finger poses: {finger_pos1},{finger_pos2}')
            # print(self.INTERP_IK, self.num_tsteps, ap)
            if self.INTERP_IK:

                new_finger_poses = [finger_pos1[0] + self.num_tsteps*ap[0], finger_pos1[1] + self.num_tsteps*ap[1], 
                                    finger_pos2[0] + self.num_tsteps*ap[2], finger_pos2[1] + self.num_tsteps*ap[3]]
                # print("target finger pose", new_finger_poses)
                found1, finger_1_angs_kegan, it1 = self.ik_f1.calculate_ik(target=new_finger_poses[:2], ee_location=None)
                found2, finger_2_angs_kegan, it12 = self.ik_f2.calculate_ik(target=new_finger_poses[2:], ee_location=None)
                action = [finger_1_angs_kegan[0],finger_1_angs_kegan[1],finger_2_angs_kegan[0],finger_2_angs_kegan[1]]
                action = clip_angs(action)
                # print(f"action: {action}, finger angles: {finger_angles}")
                # print(finger_1_angs_kegan,finger_2_angs_kegan)
                action_list = np.linspace(finger_angles,action,self.num_tsteps)
                # print(np.shape(action_list))
            else:
                for i in range(self.num_tsteps):
                    new_finger_poses = [finger_pos1[0] + ap[0], finger_pos1[1] + ap[1], finger_pos2[0] + ap[2], finger_pos2[1] + ap[3]]
                    found1, finger_1_angs_kegan, it1 = self.ik_f1.calculate_ik(target=new_finger_poses[:2], ee_location=None)
                    found2, finger_2_angs_kegan, it12 = self.ik_f2.calculate_ik(target=new_finger_poses[2:], ee_location=None)
                    action = [finger_1_angs_kegan[0],finger_1_angs_kegan[1],finger_2_angs_kegan[0],finger_2_angs_kegan[1]]
                    action = clip_angs(action)
                    action_list.append(action)
                    self.ik_f1.finger_fk.set_joint_angles(action[0:2])
                    self.ik_f2.finger_fk.set_joint_angles(action[2:4])
                    finger_pos1 = self.ik_f1.finger_fk.calculate_forward_kinematics()
                    finger_pos2 = self.ik_f2.finger_fk.calculate_forward_kinematics()
                # print(np.shape(action_list))
        else:
            finger_angles = self.gripper.get_joint_angles()
            action_list = []
            # print('current finger angles', finger_angles)
            for i in range(self.num_tsteps):
                action = ((actor_output)*self.MAX_ANGLE_CHANGE + finger_angles).tolist()
                # print('regular', action)
                action = clip_angs(action)
                # print('clipped', action)
                action_list.append(action)
                finger_angles = action
            # print('EVERYTHING IS WRONG FOR TESTING PURPOSES. IF YOU SEE THIS GO TO MULTIPROCESS CONTROL TO FIX IT')
            # temp = (actor_output*self.MAX_ANGLE_CHANGE*self.num_tsteps + finger_angles).tolist()
            # action_list = []
            # for _ in range(self.num_tsteps):
            #     action_list.append(temp.copy())
            
        # print(f'action_list {action_list}')
        return action_list, actor_output
        
    def get_network_outputs(self,state):
        return 0
    
    def get_current_cube_position(self):
        self.current_cube_pose = self.cube.get_curr_pose()

    def pre_step(self):
        self.ik_f1.finger_fk.update_angles_from_sim()
        self.ik_f2.finger_fk.update_angles_from_sim()
    
    def evaluate(self):
        if not self.train_flag:
            self.old_epsilon = self.epsilon
            self.epsilon = 0
            self.train_flag = True
        
    def train(self):
        if self.train_flag:
            self.train_flag=False
            self.epsilon = self.old_epsilon_