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
from mojograsp.simcore.DDPGfD import DDPGfD, DDPGfD_priority
from mojograsp.simcore.replay_buffer import ReplayBufferDefault, ReplayBufferDF
import torch

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
        if distance < 0.001:
            print('SUCCESS')
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


class ControllerBase:
    _sim = None

    def __init__(self, state_path=None):
        """

        :param state_path: This can be the path to a json file or a pointer to an instance of state
        """
        self.state = None

    def select_action(self):
        pass

    @staticmethod
    def create_instance(state_path, controller_type):
        """
        Create a Instance based on the controller type
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
        elif controller_type == "rl":
            return DDPGfD(state_path)
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
        self.constant_action = [1.57, 0, -1.57, 0]

    def select_action(self):
        """
        This controller is defined to open the hand.
        Thus the action will always be a constant action
        of joint position to be reached
        :return: action: action to be taken in accordance with action space
        """
        action = self.constant_action
        return action


class CloseController(ControllerBase):
    def __init__(self, state_path):
        super().__init__(state_path)
        self.constant_action = [0.7, -1.4, -0.7, 1.4]

    def select_action(self):
        """
        This controller is defined to close the hand.
        Thus the action will always be a constant action
        of joint position to be reached
        :return: action: action to be taken in accordance with action space
        """
        action = self.constant_action
        return action


class MoveController(ControllerBase):
    def __init__(self, state_path):
        super().__init__(state_path)

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
        pos = (data[4] * scale, data[5] * scale, 0.0)
        orn_eul = [0, 0, radians(data[6])]
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
        return action

class RLController(ExpertController):
    def __init__(self, gripper: TwoFingerGripper, cube: ObjectBase, data_file: str = None, replay_buffer: ReplayBufferDefault = None, args: dict = None):
        super().__init__(gripper, cube, data_file)
        if type(replay_buffer) == ReplayBufferDF:
            self.policy = DDPGfD_priority(args)
        else:
            self.policy = DDPGfD(args)
        self.replay_buffer = replay_buffer
        self.max_change = 0.1
        self.cooling_rate = 0.995
        self.rand_size = 0.1

    def get_next_action(self):

        # get current cube position
        self.get_current_cube_position()
        # get next cube position
        next_cube_position = self.get_next_cube_position()
        # get current contact points
        current_contact_points = self.get_current_contact_points()

        # if current_contact_points:
        self.get_current_cube_position()
        finger_angles = self.gripper.get_joint_angles()
        object_velocity = self.cube.get_curr_velocity()
        state = self.current_cube_pose[0] + self.current_cube_pose[1] + finger_angles + object_velocity[0] # + self.goal_position
        action = self.policy.select_action(state)
        # print('action', action)
        rand_action = self.rand_size * (np.random.rand(4) - 0.5)
        action = (action*self.max_change + finger_angles + rand_action).tolist()
        # else:
        #     print('retrying contact')
        #     self.num_contact_loss += 1
        #     self.retry_count += 1
        #     action = self.retry_contact()
        # print(action)
        return action
 
    def train_policy(self):
        # can flesh this out/try different training methods
        if type(self.replay_buffer) == ReplayBufferDF:
            if not self.replay_buffer.df_up_to_date:
                self.replay_buffer.make_DF()
        self.policy.train(None, self.replay_buffer)
        

    def update_random_size(self):
        self.rand_size = self.rand_size*self.cooling_rate

    def set_goal_position(self, position: List[float]):
        self.update_random_size()
        return super().set_goal_position(position)