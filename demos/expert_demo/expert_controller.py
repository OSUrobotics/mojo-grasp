from multiprocessing.dummy import current_process
import pybullet as p

import numpy as np

from mojograsp.simobjects.two_finger_gripper import TwoFingerGripper
from mojograsp.simobjects.object_base import ObjectBase
from typing import List
from mojograsp.simcore.jacobianIK.jacobian_IK import JacobianIK
from mojograsp.simcore.jacobianIK import matrix_helper as mh
from copy import deepcopy
import math
class ExpertController():
    # Maximum move per step
    MAX_MOVE = .01

    def __init__(self, gripper: TwoFingerGripper, cube: ObjectBase, data_file: str = None):
        self.gripper = gripper
        self.cube = cube
        self.path = data_file
        self.end_effector_links = [1, 4]

        # world coordinates
        self.goal_position = None
        # world coordinates
        self.current_cube_pose = None
        # makes sure contact isnt gone for too long
        self.num_contact_loss = 0
        self.prev_distance = 0
        self.distance_count = 0
        self.retry_count = 0

        hand_info = {"finger1": {"name": "finger0", "num_links": 2, "link_lengths": [[0, .072, 0], [0, .072, 0]]},
                         "finger2": {"name": "finger1", "num_links": 2, "link_lengths": [[0, .072, 0], [0, .072, 0]]}}
        p.resetJointState(self.gripper.id, 0, 0)
        p.resetJointState(self.gripper.id, 1, 0)
        p.resetJointState(self.gripper.id, 3, 0)
        p.resetJointState(self.gripper.id, 4, 0)
        self.ik_f1 = JacobianIK(gripper.id,deepcopy(hand_info['finger1']))
        # self.test_ik_f1 = JacobianIK(gripper.id,deepcopy(hand_info['finger1']))
        
        self.ik_f2 = JacobianIK(gripper.id,deepcopy(hand_info['finger2']))

        p.resetJointState(self.gripper.id, 0, -.725)
        p.resetJointState(self.gripper.id, 1, 1.45)
        p.resetJointState(self.gripper.id, 3, .725)
        p.resetJointState(self.gripper.id, 4, -1.45)

        p.stepSimulation()
        self.ik_f1.finger_fk.update_angles_from_sim()
        self.ik_f2.finger_fk.update_angles_from_sim()

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
            
        if self.num_contact_loss > 10:
            print('ending because contact loss')
        elif self.distance_count > 20:
            print('ending because distance count')
        elif self.check_goal() < 0.001:
            print('ending because check goal')

        # Exits if we lost contact for 5 steps, we are within .001 of our goal, or if our distance has been getting worse for 20 steps
        if self.num_contact_loss > 10 or self.check_goal() < .001 or self.distance_count > 20:
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


    def move_hand_point(self):
        #target_f1 = self.f1_direction_dict[direction]
        #target_f2 = self.f2_direction_dict[direction]
        target_f1 = self.goal_position[0:2]
        target_f2 = self.goal_position[0:2]
        
        start_f1 = self.ik_f1.finger_fk.calculate_forward_kinematics()
        start_f2 = self.ik_f2.finger_fk.calculate_forward_kinematics()
        sub_target_f1 = np.array(self.step_towards_goal(start_f1, target_f1, .003))
        sub_target_f2 = np.array(self.step_towards_goal(start_f2, target_f2, .003))

        contact_point_info1 = p.getClosestPoints(self.gripper.id, self.cube.id, .002, linkIndexA=1)
        contact_point_info2 = p.getClosestPoints(self.gripper.id, self.cube.id, .002, linkIndexA=4)
        rot = p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.cube.id)[1])
        if abs(rot[0]) > .3 or abs(rot[0]) > .3:
            print("ROTATION IS BAD")
            return
        vel = p.getBaseVelocity(self.cube.id)[0]
        if abs(vel[0]) > .55 or abs(vel[1]) > .55:
            print("velocity IS BAD")
            return

        if contact_point_info1:
            d1 = abs(contact_point_info1[-1][6][0]) - abs(contact_point_info1[-1][5][0]
                                                          ) + abs(contact_point_info1[-1][6][1]) - abs(contact_point_info1[-1][5][1])
        else:
            d1 = None
        if contact_point_info2:
            d2 = abs(contact_point_info2[-1][6][0]) - abs(contact_point_info2[-1][5][0]
                                                          ) + abs(contact_point_info2[-1][6][1]) - abs(contact_point_info2[-1][5][1])
        else:
            d2 = None

        if np.allclose(start_f1[:2], target_f1, atol=5e-4) or np.allclose(start_f2[:2], target_f2, atol=5e-3):
            print("Close enough here")
            return

        if not d2 and not d1:
            print("LOST CONTACT")
            return

        if not d1:
            print("BROKEN CONTACT 1")
            return

        if not d2:
            print("BROKEN CONTACT 2")
            return

        if contact_point_info2:
            # print("GOt HERE")

            cp1_count = 0
            t2 = mh.create_translation_matrix(contact_point_info2[-1][6])
            f2 = mh.create_transformation_matrix(
                p.getLinkState(self.gripper.id, 4)[0],
                p.getLinkState(self.gripper.id, 4)[1])
            cp2 = np.linalg.inv(f2) @ t2 @ [0, 0, 1]
            found, angles_f2, it = self.ik_f2.calculate_ik(sub_target_f2, ee_location=cp2)
            if not angles_f2:
                return
            # print(self.ik_f2.finger_fk.link_ids)
            p.setJointMotorControlArray(self.gripper.id, self.ik_f2.finger_fk.link_ids,
                                        p.POSITION_CONTROL, targetPositions=angles_f2,forces=[.3]*self.ik_f2.finger_fk.num_links)

        if contact_point_info1:
            cp2_count = 0
            t1 = mh.create_translation_matrix(contact_point_info1[-1][6])
            f1 = mh.create_transformation_matrix(
                p.getLinkState(self.gripper.id, 1)[0],
                p.getLinkState(self.gripper.id, 1)[1])
            cp1 = np.linalg.inv(f1) @ t1 @ [0, 0, 1]
            found, angles_f1, it = self.ik_f1.calculate_ik(sub_target_f1, ee_location=cp1)
            if not angles_f1:
                return
            p.setJointMotorControlArray(self.gripper.id, self.ik_f1.finger_fk.link_ids,
                                        p.POSITION_CONTROL, targetPositions=angles_f1,forces=[.3]*self.ik_f1.finger_fk.num_links )

        contact_point_info1 = p.getContactPoints(bodyA=self.gripper.id, bodyB=self.cube.id, linkIndexA=1)
        contact_point_info2 = p.getContactPoints(bodyA=self.gripper.id, bodyB=self.cube.id, linkIndexA=4)
        return [angles_f1[0],angles_f1[1],angles_f2[0],angles_f2[1]]
    
    
    def step_towards_goal(self, start_vec, end_vec, distance):
        cube_vec = self.get_cube_position()
        temp_x = end_vec[0] - cube_vec[0]
        temp_y = end_vec[1] - cube_vec[1]
        magnitude = math.sqrt((temp_x**2 + temp_y**2))
        if magnitude <= distance:
            return [end_vec[0], end_vec[1]]
        temp_x /= magnitude
        temp_y /= magnitude
        temp_x = start_vec[0] + distance*temp_x
        temp_y = start_vec[1] + distance*temp_y
        return [temp_x, temp_y]
    
    def get_cube_position(self):
        # p.changeDynamics(self.gripper.id, 1, mass=5)
        # p.changeDynamics(self.gripper.id, 3, mass=5)
        # print(p.getBasePositionAndOrientation(self.cube.id)[0])
        return p.getBasePositionAndOrientation(self.cube.id)[0]