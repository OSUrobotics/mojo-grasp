from multiprocessing.dummy import current_process
import pybullet as p

import numpy as np

from mojograsp.simobjects.two_finger_gripper import TwoFingerGripper
from mojograsp.simobjects.object_base import ObjectBase
from typing import List

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

        # Exits if we lost contact for 5 steps, we are within .001 of our goal, or if our distance has been getting worse for 20 steps
        if self.num_contact_loss > 5 or self.check_goal() < .001 or self.distance_count > 20:
            self.distance_count = 0
            self.num_contact_loss = 0
            return True
        # sets next previous distance to current distance
        self.prev_distance = self.check_goal()
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
            # print('current cube pose', self.current_cube_pose)
            # print('next cube pose', next_cube_position)
            # print('cube shift', [next_cube_position[i]-self.current_cube_pose[0][i] for i in range(3)] )
            # print('contact point 1 shift', [next_contact_points[0][0][i] - current_contact_points[0][i] for i in range(3)])
            # print('contact point 2 shift', [next_contact_points[1][0][i] - current_contact_points[1][i] for i in range(3)])
            # print('goal link positions', goal)
            # print('current link positions', )
            # # print('goal pose', self.goal_position)
            # input('how we lookin?')
        else:
            # input('retrying contact')
            self.num_contact_loss += 1
            self.retry_count+=1
            goal = self.retry_contact()

        return goal
