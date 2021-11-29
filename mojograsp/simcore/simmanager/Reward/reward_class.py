#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 10:56:40 2021
@author: orochi
"""
from . import reward_base
import os
import sys
import mojograsp
import numpy as np
import pathlib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Reward(reward_base.RewardBase):

    def __init__(self, json_path=os.path.dirname(__file__) + '/reward_demo.json'):

        """ """
        super().__init__(json_path)
        contact_reward_path = str(pathlib.Path().resolve()) + "/state_action_reward/"

        for i in self.reward_weights.keys():
            self.reward[i] = 0

            if i == 'contact':
                self.contact_state = mojograsp.state_space.StateSpace(contact_reward_path + self.reward_params['Contact_State'])

    def get_contact_reward(self):
        contact_reward = 0
        contact_points_info = self.contact_state.update()

        for i in contact_points_info:
            # i is in mm, converting to mts will make exponential extremely sensitive, which is what we want
            contact_reward += abs(1000*i)
        contact_reward = np.exp(contact_reward)
        # print("CONTACT INFO: {}\t Contact Reward: {}".format(contact_points_info, 1 * contact_reward))
        if contact_reward > 1000:
            contact_reward = -100
        else:
            contact_reward = - contact_reward/100

        return contact_reward

    def get_target_reward(self):
        curr_points = reward_base.RewardBase._sim.get_obj_curr_pose_in_start_pose()
        angle = reward_base.RewardBase._sim.objects.angle_to_horizontal
        target_pos_world, target_orn_world = reward_base.RewardBase._sim.objects.target_pose
        target_pose = reward_base.RewardBase._sim.objects.get_pose_in_start_pose(target_pos_world, target_orn_world)

        rotated_x, rotated_y = self.rotate_points(x=curr_points[0][0], y=curr_points[0][1], rad=angle)
        rot_target_x, rot_target_y = self.rotate_points(x=target_pose[0][0], y=target_pose[0][1], rad=angle)

        # print("\nOld x: {}, New x: {}, target_x: {}".format(curr_points[0][0], rotated_x, rot_target_x))
        # print("Old y: {}, New y: {}, target_y: {}\n".format(curr_points[0][1], rotated_y, rot_target_y))

        reward_x = self.get_reward_in_x(rotated_x, rot_target_x)
        reward_y = self.get_reward_in_y(rotated_y, rot_target_y)

        target_reward = -2 * reward_y + reward_x
        # print("\nTARGET REWARD: {} X: {} Y: {}\n".format(target_reward, reward_x, reward_y))
        return target_reward

    def rotate_points(self, x, y, rad):
        new_x = x * np.cos(rad) - y * np.sin(rad)
        new_y = y * np.cos(rad) + x * np.sin(rad)
        return new_x, new_y

    def get_reward_in_x(self, points_rotated, target_x):
        reward = 5 * points_rotated
        return reward

    def get_reward_in_y(self, points_rotated, target_y):
        reward = 5 * abs(points_rotated)
        return reward


if __name__ == "__main__":
    r = Reward()
    r.get_reward()
