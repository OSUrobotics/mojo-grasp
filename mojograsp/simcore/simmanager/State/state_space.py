#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 14:57:24 2021
@author: orochi
"""

import json
import time
import numpy as np
from .State_Metric.state_metric import StateMetric
from .State_Metric.all_statemetrics import *
from collections import OrderedDict


class StateSpace:
    # valid_state_names = {'Position': Position, 'Distance': Distance, 'Angle': Angle, 'Ratio': Ratio, 'Vector': Vector,
    #                      'DotProduct': DotProduct, 'StateGroup': StateGroup}
    valid_state_names = {'Angle': Angle}
    _sim = None

    def __init__(self, hand, obj, path='/Users/asar/Desktop/Grimm\'s '
                                       'Lab/Manipulation/PyBulletStuff/mojo-grasp/mojograsp/simcore/simmanager/State'
                                       '/state.json'):
        self.hand = hand
        self.object = obj
        with open(path) as f:
            json_data = json.load(f)
        self.data = OrderedDict()
        for name, value in json_data.items():
            state_name = name.split(sep='_')
            print("NAME: {} Value: {}".format(state_name, value))
            try:
                self.data[name] = eval(state_name[0] + '(value, self.hand, self.object)')
            except NameError:
                print(state_name[0], 'Invalid state name.')
        print("DICT: {}".format(self.data))

    def get_observation(self):
        obs = []
        for name, value in self.data.items():
            temp = value.get_value()
            try:
                obs.extend(temp)
            except TypeError:
                obs.extend([temp])
        return obs

    def get_specific_observation(self, keys):
        if type(keys) is str:
            keys = [keys]
        if len(keys) > 1:
            data = self.data[keys[0]].get_specific(keys[1:])
        else:
            data = self.data[keys[0]].get_value()
            # data = self.data[keys[0]].update()
        return data

    def update(self):
        for name, value in self.data.items():
            # self.data[name].update(name)
            self.data[name].update()


if __name__ == "__main__":
    pass
