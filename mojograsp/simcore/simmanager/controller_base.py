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
        This controller is defined to open the hand.
        Thus the action will always be a constant action
        of joint position to be reached
        :return: action: action to be taken in accordance with action space
        """
        action = [0, 0, 0, 0]
        return action
