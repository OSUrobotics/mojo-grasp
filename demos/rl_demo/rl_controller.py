from multiprocessing.dummy import current_process
import pybullet as p

import numpy as np

from mojograsp.simobjects.two_finger_gripper import TwoFingerGripper
from mojograsp.simobjects.object_base import ObjectBase
from typing import List
import numpy as np
import os
import pybullet as p
import pandas as pd
from math import radians
# import Markers
from mojograsp.simcore.DDPGfD import DDPGfD_priority
from mojograsp.simcore.replay_buffer import ReplayBufferDefault, ReplayBufferDF
import torch
from mojograsp.simcore.priority_replay_buffer import ReplayBufferPriority


class ExpertController():
    # Maximum move per step
    MAX_MOVE = .01

    def __init__(self, gripper: TwoFingerGripper, cube: ObjectBase, data_file: str = None):
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

    def get_current_cube_position(self):
        self.current_cube_pose = self.cube.get_curr_pose()

    def get_next_cube_position(self) -> List[float]:
        # get current (x,y)
        current_x = self.current_cube_pose[0][0]
        current_y = self.current_cube_pose[0][1]
        # get goal (x,y)
        goal_x = self.goal_position[0]
        goal_y = self.goal_position[1]
        # get distance between x1,x2 and y1,y2 and divide by our maximum movement_size to get sample size for linspace
        x_sample_size = round((max(current_x, goal_x) -
                              min(current_x, goal_x)) / self.MAX_MOVE)
        y_sample_size = round((max(current_y, goal_y) -
                              min(current_y, goal_y)) / self.MAX_MOVE)
        # if we are too close to interpolate then set next_cube_x to goal_x otherwise we use linspace
        if x_sample_size <= 1:
            next_cube_x = goal_x
        else:
            next_cube_x = np.linspace(
                current_x, goal_x, x_sample_size, endpoint=False)[1]
        # if we are too close to interpolate then set next_cube_y to goal_y otherwise we use linspace
        if y_sample_size <= 1:
            next_cube_y = goal_y
        else:
            next_cube_y = np.linspace(
                current_y, goal_y, y_sample_size, endpoint=False)[1]
        next_cube_position = [next_cube_x, next_cube_y, 0]
        return next_cube_position

    def get_current_contact_points(self) -> list:
        contact_points = []
        # left distal link
        contact_points_info_left = p.getContactPoints(
            self.cube.id, self.gripper.id, linkIndexB=self.end_effector_links[0])
        if contact_points_info_left:
            contact_points.append(contact_points_info_left[0][6])

        # right distal link
        contact_points_info_right = p.getContactPoints(
            self.cube.id, self.gripper.id, linkIndexB=self.end_effector_links[1])
        if contact_points_info_right:
            contact_points.append(contact_points_info_right[0][6])

        # if either do not have contact we return None
        if len(contact_points) < 2:
            return None
        # world coordinates
        return contact_points

    def retry_contact(self):
        # if no contact attempt to reestablish contact by moving towards cube
        location = self.current_cube_pose[0]
        next_positions = []
        # print('need to retry contact!')
        for i in range(len(self.end_effector_links)):
            distal_link = p.getLinkState(
                self.gripper.id, self.end_effector_links[i])
            distal_pos = distal_link[4]
            x_sample_size = round((max(distal_pos[0], location[0]) -
                                   min(distal_pos[0], location[0])) / self.MAX_MOVE)
            y_sample_size = round((max(distal_pos[1], location[1]) -
                                   min(distal_pos[1], location[1])) / self.MAX_MOVE)
            if x_sample_size <= 1:
                next_x = location[0]
            else:
                next_x = np.linspace(
                    distal_pos[0], location[0], x_sample_size, endpoint=False)[1]
            if y_sample_size <= 1:
                next_y = location[1]
            else:
                next_y = np.linspace(
                    distal_pos[1], location[1], y_sample_size, endpoint=False)[1]

            next_positions.append([next_x, next_y, 0])

        goal = p.calculateInverseKinematics2(bodyUniqueId=self.gripper.id,
                                             endEffectorLinkIndices=self.end_effector_links,
                                             targetPositions=next_positions)
        return goal

    def get_next_contact_points(self, current_contact_points: list, next_cube_position: list):
        next_cube_contacts_global = []
        for i in range(len(self.end_effector_links)):
            # get cube pose transform from global to local
            cube_local = p.invertTransform(
                self.current_cube_pose[0], self.current_cube_pose[1])
            # take cube pose and multiply it by current contact points to get them in local frame
            cube_contacts_local = p.multiplyTransforms(
                cube_local[0], cube_local[1], current_contact_points[i], self.current_cube_pose[1])
            # get contact contact points for next cube position using current contact points, returns in global frame
            # ORIENTATION MAY BE IMPORTANT HERE
            next_cube_contacts_global.append(p.multiplyTransforms(next_cube_position, [0, 0, 0, 1],
                                                                  cube_contacts_local[0], cube_contacts_local[1]))
        return next_cube_contacts_global

    def get_next_link_positions(self, current_contact_points: list, next_contact_points: list):
        next_link_positions_global = []
        for i in range(len(self.end_effector_links)):
            # gets contact points into local frame
            contacts_local = p.invertTransform(
                current_contact_points[i], self.current_cube_pose[1])
            # get distal link information
            distal_link = p.getLinkState(
                self.gripper.id, self.end_effector_links[i])
            # get current contact points in relation to distal link
            distal_contacts_local = p.multiplyTransforms(
                contacts_local[0], contacts_local[1], distal_link[4], distal_link[5])
            # get next contact points in global coordinates
            next_link_positions_global.append(p.multiplyTransforms(
                next_contact_points[i][0], next_contact_points[i][1], distal_contacts_local[0], distal_contacts_local[1])[0])

        goal = p.calculateInverseKinematics2(bodyUniqueId=self.gripper.id,
                                             endEffectorLinkIndices=self.end_effector_links,
                                             targetPositions=next_link_positions_global)
        return goal

    def set_goal_position(self, position: List[float]):
        # world coordinates
        self.goal_position = position

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

    def get_next_action(self):
        # get current cube position
        self.get_current_cube_position()
        # get next cube position
        next_cube_position = self.get_next_cube_position()
        # get current contact points
        current_contact_points = self.get_current_contact_points()

        if current_contact_points:
            self.num_contact_loss = 0
            # find next contact points
            next_contact_points = self.get_next_contact_points(
                current_contact_points=current_contact_points, next_cube_position=next_cube_position)
            # get goal link positions
            goal = self.get_next_link_positions(
                current_contact_points=current_contact_points, next_contact_points=next_contact_points)

        else:
            # input('retrying contact')
            self.num_contact_loss += 1
            self.retry_count+=1
            goal = self.retry_contact()

        return goal

class RLController(ExpertController):
    def __init__(self, gripper: TwoFingerGripper, cube: ObjectBase, data_file: str = None, replay_buffer: ReplayBufferDefault = None, args: dict = None, tbname=None):
        super().__init__(gripper, cube, data_file)
        if type(replay_buffer) == ReplayBufferDF or type(replay_buffer) == ReplayBufferPriority:
            self.policy = DDPGfD_priority(args, tbname)
        self.replay_buffer = replay_buffer
        try:
            self.max_change = 0.1
            self.cooling_rate = 0.995
            self.rand_portion = np.array([0,0,0,0])
            self.final_reward = 0
            self.epsilon = 0.7
        except:
            print('no config given, using baseline parameters')
            self.max_change = 0.1
            self.cooling_rate = args['edecay']
            # self.rand_size = 0.1
            self.rand_portion = np.array([0,0,0,0])
            self.final_reward = 0
            self.epsilon = args['epsilon']
        self.rand_episode = np.random.rand() < self.epsilon


    def get_next_action(self):

        # get current cube position
        self.get_current_cube_position()

        finger_angles = self.gripper.get_joint_angles()
        # object_velocity = self.cube.get_curr_velocity()
        # f1 = p.getClosestPoints(self.cube.id, self.gripper.id, 10, -1, 1, -1)[0]
        # f2 = p.getClosestPoints(self.cube.id, self.gripper.id, 10, -1, 3, -1)[0]
        # finger_pos1 = p.getLinkState(self.gripper.id, 1)
        # finger_pos2 = p.getLinkState(self.gripper.id, 3)
        # For new Configuration
        finger_pos1 = p.getLinkState(self.gripper.id, 2)
        finger_pos2 = p.getLinkState(self.gripper.id, 5)
        # f1_loc = f1[5]
        # f2_loc = f1[5]
        # f1_dist = f1[8]
        # f2_dist = f2[8]
        # state = self.current_cube_pose[0] + self.current_cube_pose[1] + finger_angles + object_velocity[0] + [f1_dist, f2_dist] + list(f1_loc) + list(f2_loc)# + self.goal_position
        # state = self.current_cube_pose[0][0:2] + finger_angles + object_velocity[0][0:2] + [f1_dist, f2_dist] + list(f1_loc)[0:2] + list(f2_loc)[0:2] + self.goal_position[0:2]
        
        state = self.current_cube_pose[0][0:2] + list(finger_pos1[0])[0:2] + list(finger_pos2[0])[0:2] + self.goal_position[0:2]
        
        if not self.rand_episode:
            actor_portion = self.policy.select_action(state)
            
            action = ((actor_portion+(np.random.rand(4)-0.5)/2 * self.epsilon)*self.max_change + finger_angles).tolist()
        else:
            actor_portion = self.rand_portion + (np.random.rand(4)-0.5)/2
            action = (self.max_change*(actor_portion) + finger_angles).tolist()
        
        return action, actor_portion
    
    def get_next_IK_action(self):
        # get current cube position
        self.get_current_cube_position()

        finger_angles = self.gripper.get_joint_angles()
        # object_velocity = self.cube.get_curr_velocity()
        # f1 = p.getClosestPoints(self.cube.id, self.gripper.id, 10, -1, 1, -1)[0]
        # f2 = p.getClosestPoints(self.cube.id, self.gripper.id, 10, -1, 3, -1)[0]
        # finger_pos1 = p.getLinkState(self.gripper.id, 1)
        # finger_pos2 = p.getLinkState(self.gripper.id, 3)
        # For new Configuration
        finger_pos1 = p.getLinkState(self.gripper.id, 2)
        finger_pos2 = p.getLinkState(self.gripper.id, 5)
        # f1_loc = f1[5]
        # f2_loc = f1[5]
        # f1_dist = f1[8]
        # f2_dist = f2[8]
        # state = self.current_cube_pose[0] + self.current_cube_pose[1] + finger_angles + object_velocity[0] + [f1_dist, f2_dist] + list(f1_loc) + list(f2_loc)# + self.goal_position
        # state = self.current_cube_pose[0][0:2] + finger_angles + object_velocity[0][0:2] + [f1_dist, f2_dist] + list(f1_loc)[0:2] + list(f2_loc)[0:2] + self.goal_position[0:2]
        
        state = self.current_cube_pose[0][0:2] + list(finger_pos1[0])[0:2] + list(finger_pos2[0])[0:2] + self.goal_position[0:2]
        
        if not self.rand_episode:           
            actor_portion = self.policy.select_action(state) * 0.01
            # print(finger_deltas)
            new_finger_poses = [finger_pos1[0][0] + actor_portion[0],finger_pos1[0][1] + actor_portion[1],finger_pos2[0][0] + actor_portion[2],finger_pos2[0][1] + actor_portion[3]]
            # action = (action*self.max_change + finger_angles).tolist()
            # print(new_finger_poses)
            # p.calculateInverseKinematic
            finger_1_angs = p.calculateInverseKinematics(self.gripper.id,2,[new_finger_poses[0], new_finger_poses[1], finger_pos1[0][2]])
            finger_2_angs = p.calculateInverseKinematics(self.gripper.id,5,[new_finger_poses[2], new_finger_poses[3], finger_pos2[0][2]])
            # print(np.array(finger_1_angs) - np.array(finger_angles), np.array(finger_2_angs) - np.array(finger_angles))
            action = [finger_1_angs[0],finger_1_angs[1],finger_2_angs[2],finger_2_angs[3]]
            # print('current finger poses', finger_pos1[0][0:2], finger_pos2[0][0:2])
            # print('desired new finger poses', new_finger_poses)
            # print('ik angle change', np.array(action) - np.array(finger_angles))
        else:
            
            actor_portion = (self.rand_portion + np.random.rand(4)-0.5) * 0.01
            new_finger_poses = [finger_pos1[0][0] + actor_portion[0],finger_pos1[0][1] + actor_portion[1],finger_pos2[0][0] + actor_portion[2],finger_pos2[0][1] + actor_portion[3]]
            # p.calculateInverseKinematic
            finger_1_angs = p.calculateInverseKinematics(self.gripper.id,2,[new_finger_poses[0], new_finger_poses[1], finger_pos1[0][2]])
            finger_2_angs = p.calculateInverseKinematics(self.gripper.id,5,[new_finger_poses[2], new_finger_poses[3], finger_pos2[0][2]])
            action = [finger_1_angs[0],finger_1_angs[1],finger_2_angs[2],finger_2_angs[3]]
            # print('random angle change',np.array(action) - np.array(finger_angles))
        
        return action, actor_portion
    
    def get_network_outputs(self):
        self.get_current_cube_position()
        # get next cube position
        # finger_angles = self.gripper.get_joint_angles()
        # object_velocity = self.cube.get_curr_velocity()
        # f1 = p.getClosestPoints(self.cube.id, self.gripper.id, 10, -1, 1, -1)[0]
        # f2 = p.getClosestPoints(self.cube.id, self.gripper.id, 10, -1, 1, -1)[0]
        # finger_pos1 = p.getLinkState(self.gripper.id, 1)
        # finger_pos2 = p.getLinkState(self.gripper.id, 3)
        # For new Configuration
        finger_pos1 = p.getLinkState(self.gripper.id, 2)
        finger_pos2 = p.getLinkState(self.gripper.id, 5)
        # f1_loc = f1[5]
        # f2_loc = f1[5]
        
        # f1_dist = f1[8]
        # f2_dist = f2[8]
        # state = self.current_cube_pose[0] + self.current_cube_pose[1] + finger_angles + object_velocity[0] + [f1_dist, f2_dist] + list(f1_loc) + list(f2_loc)# + self.goal_position

        # state = self.current_cube_pose[0][0:2] + finger_angles + object_velocity[0][0:2] + [f1_dist, f2_dist] + list(f1_loc)[0:2] + list(f2_loc)[0:2] + self.goal_position[0:2]
        
        state = self.current_cube_pose[0][0:2] + list(finger_pos1[0])[0:2] + list(finger_pos2[0])[0:2] + self.goal_position[0:2]
        
        action = self.policy.select_action(state)
        
        critic_response = self.policy.grade_action(state, action)
        
        save_dict = {'actor_output' : action.tolist(), 'critic_output' : critic_response[0]}
        
        return save_dict

    def train_policy(self):
        # can flesh this out/try different training methods
        if type(self.replay_buffer) == ReplayBufferDF:
            if not self.replay_buffer.df_up_to_date:
                self.replay_buffer.make_DF()
        self.policy.train(self.replay_buffer)
        
    def exit_condition(self, remaining_tstep=0):
        # checks if we are getting further from goal or closer
        goal_dist = self.check_goal()
        if self.prev_distance <= goal_dist:
            self.distance_count += 1
        else:
            self.distance_count = 0
            

        # Exits if we lost contact for 5 steps, we are within .002 of our goal, or if our distance has been getting worse for 20 steps
        if goal_dist < .002:
            self.distance_count = 0
            self.final_reward = 1
            print('exiting in rl controller because we reached the goal')
            return True
        
        if goal_dist > 0.2:
            self.distance_count = 0
            print('exiting in rl controller because we were 0.2 m away')
            return True
        
        # if self.distance_count > 40:
        #     vel = np.array(self.cube.get_curr_velocity()[0])
        #     pos = np.array(self.cube.get_curr_pose()[0])
        #     final_pos = pos + vel * remaining_tstep
        #     self.final_reward = -np.sqrt((self.goal_position[0] - final_pos[0])**2 +
        #                                 (self.goal_position[1] - final_pos[1])**2) * remaining_tstep
        #     print('exiting in rl controller because distance count is > 40', self.final_reward)
        #     self.distance_count = 0
        #     return True
        # sets next previous distance to current distance
        self.prev_distance = self.check_goal().copy()
        self.final_reward = 0
        return False
    
    
    def update_random_size(self):
        # self.rand_size = self.rand_size*self.cooling_rate
        # self.rand_portion = self.rand_size * (np.random.rand(4) - 0.5)
        
        self.rand_portion = 0.5 * (np.random.rand(4) - 0.5)
        # print(self.rand_portion)
        self.epsilon = self.epsilon * self.cooling_rate
        self.rand_episode = np.random.rand() < self.epsilon
        if self.rand_episode:
            print('NEXT EPISODE WILL BE RANDOM')
        else:
            print('NEXT EPISODE WILL BE POLICY BASED')

    def set_goal_position(self, position: List[float]):
        self.update_random_size()
        return super().set_goal_position(position)
    
