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
        print('epsilon and edecay', self.epsilon, self.COOLING_RATE)
        self.rand_episode = np.random.rand() < self.epsilon
        self.useIK = args['action']=="Finger Tip Position"
        # self.eval_flag = False
        
        self.gripper = gripper
        self.cube = cube
        self.path = data_file
        self.end_effector_links = [1, 3]

        # world coordinates
        self.goal_position = None
        # world coordinates
        self.current_cube_pose = None
        # makes sure contact isnt gone for too long
        self.num_contact_loss = 0
        self.prev_distance = 0
        self.distance_count = 0
        self.retry_count = 0
        keys = hand_type.split('_')
        if keys[1] == '50.50':
            f1 = {"name": "finger0", "num_links": 2, "link_lengths": [[0, .072, 0], [0, .072, 0]]}
        elif keys[1] == "70.30":
            f1 = {"name": "finger0", "num_links": 2, "link_lengths": [[0, .1008, 0], [0, .0432, 0]]}
        if keys[2] == '50.50':
            f2 = {"name": "finger1", "num_links": 2, "link_lengths": [[0, .072, 0], [0, .072, 0]]}
        elif keys[2] == "70.30":
            f2 = {"name": "finger1", "num_links": 2, "link_lengths": [[0, .1008, 0], [0, .0432, 0]]}
        hand_info = {"finger1": f1,"finger2": f2}
        self.p.resetJointState(self.gripper.id, 0, 0)
        self.p.resetJointState(self.gripper.id, 1, 0)
        self.p.resetJointState(self.gripper.id, 3, 0)
        self.p.resetJointState(self.gripper.id, 4, 0)
        self.ik_f1 = JacobianIK(self.p, gripper.id,deepcopy(hand_info['finger1']))
        
        self.ik_f2 = JacobianIK(self.p, gripper.id,deepcopy(hand_info['finger2']))

        if keys[1] == '50.50':
            self.p.resetJointState(self.gripper.id, 0, -.725)
            self.p.resetJointState(self.gripper.id, 1, 1.45)
        elif keys[1] == "70.30":
            self.p.resetJointState(self.gripper.id, 0, -.5)
            self.p.resetJointState(self.gripper.id, 1, 1.5)
        if keys[2] == '50.50':
            self.p.resetJointState(self.gripper.id, 3, .725)
            self.p.resetJointState(self.gripper.id, 4, -1.45)
        elif keys[2] == "70.30":
            self.p.resetJointState(self.gripper.id, 3, .5)
            self.p.resetJointState(self.gripper.id, 4, -1.5)
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
        if self.useIK:
            finger_pos1 = self.p.getLinkState(self.gripper.id, 2) #RIGHT FINGER
            finger_pos2 = self.p.getLinkState(self.gripper.id, 5) #LEFT FINGER
            finger_pos1 = finger_pos1[0]
            finger_pos2 = finger_pos2[0]
            finger_angles = self.gripper.get_joint_angles()
            ap = actor_output * self.MAX_DISTANCE_CHANGE
            action_list = []
            # print(f'actor_output: {actor_output}, finger poses: {finger_pos1},{finger_pos2}')
            if self.INTERP_IK:
                new_finger_poses = [finger_pos1[0] + self.num_tsteps*ap[0], finger_pos1[1] + self.num_tsteps*ap[1], 
                                    finger_pos2[0] + self.num_tsteps*ap[2], finger_pos2[1] + self.num_tsteps*ap[3]]
                found1, finger_1_angs_kegan, it1 = self.ik_f1.calculate_ik(target=new_finger_poses[:2], ee_location=None)
                found2, finger_2_angs_kegan, it12 = self.ik_f2.calculate_ik(target=new_finger_poses[2:], ee_location=None)
                action = [finger_1_angs_kegan[0],finger_1_angs_kegan[1],finger_2_angs_kegan[0],finger_2_angs_kegan[1]]
                action = clip_angs(action)
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
            # print(actor_output, finger_angles)
            for i in range(self.num_tsteps):
                action = ((actor_output)*self.MAX_ANGLE_CHANGE + finger_angles).tolist()
                action = clip_angs(action)
                action_list.append(action)
                finger_angles = action
        # print(f'action_list {action_list}')
        return action_list, actor_output
        
    def get_network_outputs(self,state):
        return 0
    
    def get_current_cube_position(self):
        self.current_cube_pose = self.cube.get_curr_pose()

    def set_goal_position(self, position: List[float]):
        # world coordinates
        self.goal_position = position
        self.ik_f1.finger_fk.update_angles_from_sim()
        self.ik_f2.finger_fk.update_angles_from_sim()

    def check_goal(self):
        # Finds distance between current cube position and goal position
        distance = np.sqrt((self.goal_position[0] - self.current_cube_pose[0][0])**2 +
                           (self.goal_position[1] - self.current_cube_pose[0][1])**2)
        return distance

    def exit_condition(self):
        # checks if we are getting further from goal or closer
        if self.prev_distance < self.check_goal():
            self.distance_count += 1
        else:
            self.distance_count = 0

        # Exits if we lost contact for 5 steps, we are within .002 of our goal, or if our distance has been getting worse for 20 steps
        if self.num_contact_loss > 5 or self.check_goal() < .002 or self.distance_count > 20:
            self.distance_count = 0
            self.num_contact_loss = 0
            return True
        # sets next previous distance to current distance
        self.prev_distance = self.check_goal().copy()
        return False

    def exit_condition(self, remaining_tstep=0):
        # checks if we are getting further from goal or closer
        goal_dist = self.check_goal()
        if self.prev_distance <= goal_dist:
            self.distance_count += 1
        else:
            self.distance_count = 0
            
        if goal_dist < .002:
            self.distance_count = 0
            self.final_reward = 1
            print('exiting in rl controller because we reached the goal')
            return True
        
        if goal_dist > 0.2:
            self.distance_count = 0
            print('exiting in rl controller because we were 0.2 m away')
            return True
        
        self.prev_distance = self.check_goal().copy()
        self.final_reward = 0
        return False

    def set_goal_position(self, position: List[float]):
        self.goal_position = position
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
            self.epsilon = self.old_epsilon