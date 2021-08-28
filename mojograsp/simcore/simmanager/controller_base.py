#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 11:14:16 2021
@author: orochi
"""
import numpy as np
import os
import mojograsp


class ControllerBase:
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


class GenericController(ControllerBase):
    def __init__(self, controller_type, state_path=None):
        super().__init__()
        if controller_type == "open":
            self.controller = OpenController(state_path)
        elif controller_type == "close":
            self.controller = CloseController(state_path)
        elif controller_type == "PID move":
            self.controller = PIDMoveController(state_path)
        else:
            self.controller = controller_type
        try:
            self.select_action = self.controller.select_action
        except AttributeError:
            raise AttributeError('Invalid controller type. '
                                 'Valid controller types are: "open", "close" and "PID move"')


class OpenController(ControllerBase):
    def __init__(self, state_path):
        super().__init__()

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
        super().__init__()

    def select_action(self):
        """
        This controller is defined to open the hand.
        Thus the action will always be a constant action
        of joint position to be reached
        :return: action: action to be taken in accordance with action space
        """
        action = [0, 0, 0, 0]
        return action


class PIDMoveController(ControllerBase):
#State space needs to include next pose to go to, contact points, probably a sim pointer? for access to hand joint angle numbers, etc
    def __init__(self, state_path):
        super().__init__()
        self.timestep = 0

    def select_action(self):
        """ Move fingers at a constant speed, return action """

        bell_curve_velocities = [0.202, 0.27864, 0.35046, 0.41696, 0.47814, 0.534, 0.58454, 0.62976, 0.66966, 0.70424,
                                 0.7335, 0.75744, 0.77606, 0.78936, 0.79734, 0.8, 0.79734, 0.78936, 0.77606, 0.75744,
                                 0.7335, 0.70424, 0.66966, 0.62976, 0.58454, 0.534, 0.47814, 0.41696, 0.35046, 0.27864,
                                 0.2015]

        # Determine the finger velocities by increasing and decreasing the values with a constant acceleration
        finger_velocity = bell_curve_velocities[self.timestep]
        self.timestep += 1
        # By default, close all fingers at a constant speed
        action = np.array([finger_velocity, finger_velocity, finger_velocity])

        # If ready to lift, set fingers to constant lifting velocities
        if self.lift_check is True:
            action = np.array(
                [self.const_velocities["finger_lift_velocity"], self.const_velocities["finger_lift_velocity"],
                 self.const_velocities["finger_lift_velocity"]])
        # print("TS: {} action: {}".format(timestep, action))

        return action
