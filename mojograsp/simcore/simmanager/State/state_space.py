#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 14:57:24 2021
@author: orochi
"""

import json
import time
import numpy as np
from mojograsp.simcore.simmanager.State.State_Metric.state_metric import StateMetricAngle
# from mojograsp.simcore.simmanager.State.State_Metric.state_metric import Angle_JointState
from collections import OrderedDict


class StateSpace:
    # valid_state_names = {'Position': Position, 'Distance': Distance, 'Angle': Angle, 'Ratio': Ratio, 'Vector': Vector,
    #                      'DotProduct': DotProduct, 'StateGroup': StateGroup}
    valid_state_names = {'Angle': StateMetricAngle}
    # valid_state_names = {'Angle': Angle_JointState}
    _sim = None

    def __init__(self, path='/Users/asar/Desktop/Grimm\'s '
                                       'Lab/Manipulation/PyBulletStuff/mojo-grasp/mojograsp/simcore/simmanager/State'
                                       '/state.json'):
        print('path',path)
        with open(path) as f:
            json_data = json.load(f)
        self.data = OrderedDict()
        for name, value in json_data.items():
            state_name = name.split(sep='_')
            try:
                print(state_name[0])
                self.data[name] = StateSpace.valid_state_names[state_name[0]](value)
            except NameError:
                print(state_name[0],'Invalid state name. Valid state names are', [name for name in
                                                                           StateSpace.valid_state_names.keys()])

    def get_obs(self):
        #self.update()
        arr = []
        for name, value in self.data.items():
            temp = value.get_value()
            try:
                arr.extend(temp)
            except TypeError:
                arr.extend([temp])
        return arr

    def get_value(self, keys):
        if type(keys) is str:
            keys = [keys]
        if len(keys) > 1:
            data = self.data[keys[0]].get_specific(keys[1:])
        else:
            data = self.data[keys[0]].get_value()
        return data

    def update(self):
        for name, value in self.data.items():
            print("KEY IS: {}, {}".format(name, self.data[name]))
            self.data[name].update(name)
        return self.get_obs()

    # def __init__(self, hand, obj, path='/Users/asar/Desktop/Grimm\'s '
    #                                    'Lab/Manipulation/PyBulletStuff/mojo-grasp/mojograsp/simcore/simmanager/State'
    #                                    '/state.json'):
    #     self.hand = hand
    #     self.object = obj
    #     with open(path) as f:
    #         json_data = json.load(f)
    #     self.data = OrderedDict()
    #     for name, value in json_data.items():
    #         state_name = name.split(sep='_')
    #         print("NAME: {} Value: {}".format(state_name, value))
    #         try:
    #             self.data[name] = eval(state_name[0] + '(value, self.hand, self.object)')
    #         except NameError:
    #             print(state_name[0], 'Invalid state name.')
    #     print("DICT: {}".format(self.data))
    #
    # def get_observation(self):
    #     obs = []
    #     for name, value in self.data.items():
    #         temp = value.get_value()
    #         try:
    #             obs.extend(temp)
    #         except TypeError:
    #             obs.extend([temp])
    #     return obs
    #
    # def get_specific_observation(self, keys):
    #     if type(keys) is str:
    #         keys = [keys]
    #     if len(keys) > 1:
    #         data = self.data[keys[0]].get_specific(keys[1:])
    #     else:
    #         data = self.data[keys[0]].get_value()
    #         # data = self.data[keys[0]].update()
    #     return data
    #
    # def update(self):
    #     for name, value in self.data.items():
    #         # self.data[name].update(name)
    #         self.data[name].update()
    #

if __name__ == "__main__":
    try_state = StateSpace()

    try_state.update()
    pass
