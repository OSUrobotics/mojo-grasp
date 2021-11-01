#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 11:14:16 2021
@author: orochi
"""
import numpy as np
import os
import mojograsp
import pybullet as p
import pandas as pd
from math import radians
#import Markers


class ControllerBase:
    _sim = None

    def __init__(self, state_path=None):
        """

        :param state_path: This can be the path to a json file or a pointer to an instance of state
        """
        if '.json' in state_path:
            self.state = mojograsp.state_space.StateSpace(path=state_path)
        else:
            self.state = state_path

    def select_action(self):
        pass

    @staticmethod
    def create_instance(state_path, controller_type):
        """
        Create a Instance based on the contorller type
        :param state_path: is the json file path or instance of state space
        :param controller_type: string type, name of controller defining which controller instance to create
        :return: An instance based on the controller type
        """
        if controller_type == "open":
            return OpenController(state_path)
        elif controller_type == "close":
            return CloseController(state_path)
        elif controller_type == "move":
            return MoveController(state_path)
        else:
            controller = controller_type
            try:
                controller.select_action
            except AttributeError:
                raise AttributeError('Invalid controller type. '
                                     'Valid controller types are: "open", "close" and "PID move"')
            return controller


class OpenController(ControllerBase):
    def __init__(self, state_path):
        super().__init__(state_path)

    def select_action(self):
        """
        This controller is defined to open the hand.
        Thus the action will always be a constant action
        of joint position to be reached
        :return: action: action to be taken in accordance with action space
        """
        action = [1.57, 0, -1.57, 0]
        return action


class CloseController(ControllerBase):
    def __init__(self, state_path):
        super().__init__(state_path)

    def select_action(self):
        """
        This controller is defined to close the hand.
        Thus the action will always be a constant action
        of joint position to be reached
        :return: action: action to be taken in accordance with action space
        """
        action = [0.78, -1.65, -0.78, 1.65]
        return action


class MoveController(ControllerBase):
    def __init__(self, state_path):
        super().__init__(state_path)
        self.dir = 'c'
        #TODO: Change to non hardcoded path :)
        self.filename = "/home/keegan/mojo/mojo2/mojo-grasp/demos/full_rl_demo/asterisk_test_data_for_anjali/trial_paths/not_normalized/sub1_2v2_{}_n_1.csv".format(self.dir)
        self.object_poses_expert = self.extract_data_from_file()
        self.iterator = 0
        self.data_len = len(self.object_poses_expert)
        self.data_over = False

    def extract_data_from_file(self):
        """
        Read in csv file  containing  information  of human studies as a panda dataframe.
        Convert it  to numpy arrays
        Format: Start pos of hand is origin
        x,y,rmag,f_x,f_y,f_rmag
        Note: c dir is +x; a dir is +y [from humanstudy data]
        :param filename: Name of file containing human data
        :return: numpy  array containing the information from the file
        """
        df = pd.read_csv(self.filename)
        df.drop(df.columns[df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
        # print("Head of file: \n", df.head())
        data = df.to_numpy()
        return data

    def get_next_line(self):
        next_pose = self.object_poses_expert[self.iterator]
        self.iterator += 1
        if self.iterator >= self.data_len:
            self.data_over = True
        return next_pose

    @staticmethod
    def _convert_data_to_pose(data, scale=0.001):
        """
        TODO: Change name to better represent what's happening here "convert_data_to_pose"?
        Get the next pose values from the file in the proper pose format
        :param data: Data line from file as a list [x, y, rotx,  f_x,f_y,f_rot_mag]
        :return:
        """
        pos = (data[4] * scale, 0.0, data[5] * scale)
        orn_eul = [0, radians(data[6]), 0]
        orn = p.getQuaternionFromEuler(orn_eul)
        return pos, orn

    def _get_origin_cube(self, data):
        """
        TODO: make this private?
        Gets the next pose of the  cube in world coordinates
        :param cube:
        :param data:
        :return:
        """
        next_pos, next_orn = self._convert_data_to_pose(data)
        T_origin_next_pose_cube = p.multiplyTransforms(self._sim.objects.start_pos[self._sim.objects.id][0],
                                                       self._sim.objects.start_pos[self._sim.objects.id][1], next_pos,
                                                       next_orn)
        return T_origin_next_pose_cube

    def get_contact_points(self, cube_id):
        """
        Get the contact points between object passed in (cube) and gripper
        If no  contact, returns None
        :param cube_id:
        :return: [contact_points_left, contact_points_right] left and right finger contacts
        """
        contact_points = []
        for i in range(0, len(self._sim.hand.end_effector_indices)):
            contact_points_info = p.getContactPoints(cube_id, self._sim.hand.id,
                                                     linkIndexB=self._sim.hand.end_effector_indices[i])
            try:
                contact_points.append(contact_points_info[0][6])
            except IndexError:
                contact_points.append(None)

        return contact_points

    def maintain_contact(self):
        # print("No Contact")
        target_pos = []
        go_to, _ = self._sim.objects.get_curr_pose()
        for j in range(0, len(self._sim.hand.end_effector_indices)):
            target_pos.append(go_to)
        return target_pos

    def get_origin_cp(self, i, cube, T_cube_origin, T_origin_nextpose_cube, curr_contact_points):
        """
        TODO: make this private?
        :return:
        """
        pos, curr_obj_orn = self._sim.objects.get_curr_pose()
        # curr_contact_points[i] = 0,0,0
        # print("Current Contacts: {}\nCurrent Orientation: {}".format(curr_contact_points[i], curr_obj_orn))
        T_cube_cp = p.multiplyTransforms(T_cube_origin[0], T_cube_origin[1], curr_contact_points[i], curr_obj_orn)
        T_origin_new_cp = p.multiplyTransforms(T_origin_nextpose_cube[0], T_origin_nextpose_cube[1],
                                               T_cube_cp[0], T_cube_cp[1])

        return T_origin_new_cp

    def get_origin_links(self, i, j, T_origin_newcontactpoints_pos, T_origin_newcontactpoints_orn, curr_contact_points):
        """
        TODO: make this private?
        :param i:
        :param T_origin_newcontactpoints:
        :return:
        """
        _, curr_obj_orn = self._sim.objects.get_curr_pose()
        T_cp_origin = p.invertTransform(curr_contact_points[i], curr_obj_orn)
        link = p.getLinkState(self._sim.hand.id, j)
        distal_pos = link[4]
        distal_orn = link[5]
        T_cp_link = p.multiplyTransforms(T_cp_origin[0], T_cp_origin[1], distal_pos, distal_orn)
        T_origin_nl = p.multiplyTransforms(T_origin_newcontactpoints_pos[i], T_origin_newcontactpoints_orn[i],
                                           T_cp_link[0], T_cp_link[1])

        return T_origin_nl

    def _get_pose_in_world_origin_expert(self, data):
        """
        TODO: make this private?
        Gets the new contact points in world coordinates
        :param cube: instance of object in scene class(object to move)
        :param data: line in file of human data [x,y,rmag,f_x,f_y,f_rot_mag]
        :return: list T_origin_newcontactpoints: next contact points in world coordinates for left and right
        """

        T_origin_nextpose_cube = self._get_origin_cube(data)
        curr_contact_points = self.get_contact_points(self._sim.objects.id)
        if None in curr_contact_points:
            return [None, None, self.maintain_contact(), None, None]
        obj_pos, obj_orn = self._sim.objects.get_curr_pose()
        T_cube_origin = p.invertTransform(obj_pos, obj_orn)
        T_origin_new_cp_pos = []
        T_origin_new_cp_orn = []
        T_origin_new_link_pos = []
        T_origin_new_link_orn = []
        for i in range(0, len(self._sim.hand.end_effector_indices)):
            T_origin_new_cp = self.get_origin_cp(i, self._sim.objects.id, T_cube_origin, T_origin_nextpose_cube,
                                                 curr_contact_points)
            # print("Contact")
            T_origin_new_cp_pos.append(T_origin_new_cp[0])
            T_origin_new_cp_orn.append(T_origin_new_cp[1])
            T_origin_nl = self.get_origin_links(i, self._sim.hand.end_effector_indices[i], T_origin_new_cp_pos,
                                                T_origin_new_cp_orn, curr_contact_points)
            T_origin_new_link_pos.append(T_origin_nl[0])
            T_origin_new_link_orn.append(T_origin_nl[1])

        return [T_origin_new_cp_pos, T_origin_new_cp_orn, T_origin_new_link_pos, T_origin_new_link_orn,
         T_origin_nextpose_cube]
        # return T_origin_new_link_pos

    def select_action(self):
        """
        This controller is designed ot move an object along a certain path
        :return: action
        """
        cube_next_pose = self.get_next_line()
        # print("Cube Pose: {}".format(cube_next_pose))
        next_info = self._get_pose_in_world_origin_expert(cube_next_pose)
        # print(next_info)
        next_contact_points = next_info[2]
        action = p.calculateInverseKinematics2(bodyUniqueId=self._sim.hand.id,
                                               endEffectorLinkIndices=self._sim.hand.end_effector_indices,
                                               targetPositions=next_contact_points)
        # if next_info[0] is not None:
        #     Markers.Marker().set_marker_pose(next_info[0][0])
        #     Markers.Marker().set_marker_pose(next_info[0][1])
        return action
