#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 14:57:24 2021
@author: orochi
"""
import time
import numpy as np
import re
from pathlib import Path
import os
import sys
import pybullet as p
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mojograsp.simcore.datacollection.stats_tracker_base import *
from collections import OrderedDict
from mojograsp.simcore.simmanager.State.State_Metric.state_metric_base import StateMetricBase


class StateMetricPyBullet(StateMetricBase):

    def get_value(self):
        return self.data.value

    def get_index_from_keys(self, keys):
        pass


class StateMetricAngle(StateMetricPyBullet):
    def update(self, keys):
        print("BEFORE Here Angle: {}, \nKeys: ".format(self.data, keys))
        joint_indices = self.get_index_from_keys(keys)
        curr_joint_angles = StateMetricBase._sim.get_hand_curr_joint_angles(joint_indices)
        self.data.set_value(curr_joint_angles)
        print("AFTER Here Angle: {}".format(self.data))


class StateMetricPosition(StateMetricPyBullet):
    def update(self, keys):
        print("KEYS: {}".format(keys))
        if 'F1' in keys:
            pass

        elif 'F2' in keys:
            pass

        elif 'Palm' in keys:
            object_list_index = 0
            curr_pose = StateMetricBase._sim.get_obj_curr_pose(object_list_index)[0]
            print("POSE:", curr_pose)
            print("DATA:", self.data)
            self.data.set_value(curr_pose)


class StateMetricVector(StateMetricPyBullet):
    pass


class StateMetricRatio(StateMetricPyBullet):
    pass


class StateMetricDistance(StateMetricPyBullet):
    def update(self, keys):
        if 'ObjSize' in keys:
            print("KEYS: {}".format(keys))
            dimensions = p.getVisualShapeData(StateMetricBase._sim.objects[0].id)[0][3]
            print("DIMENSIONS: {}".format(dimensions))
            self.data.set_value(dimensions)


class StateMetricDotProduct(StateMetricPyBullet):
    pass


class StateMetricGroup(StateMetricPyBullet):
    valid_state_names = {'Position': StateMetricPosition, 'Distance': StateMetricDistance, 'Angle': StateMetricAngle,
                         'Ratio': StateMetricRatio, 'Vector': StateMetricVector, 'DotProduct': StateMetricDotProduct,
                         'StateGroup':'StateMetricGroup'}

    def __init__(self, data_structure):
        super().__init__(data_structure)
        self.data = OrderedDict()
        for name, value in data_structure.items():
            state_name = name.split('_')
            try:
                print('state name',state_name[0])
                self.data[name] = StateMetricGroup.valid_state_names[state_name[0]](value)
            except TypeError:
                self.data[name] = StateMetricGroup(value)
            except KeyError:
                print('Invalid state name. Valid state names are', [name
                          for name in StateMetricGroup.valid_state_names.keys()])

    def update(self, keys):
        arr = []
        print("Here Metric Group: {}\nKeys: {}".format(self.data, keys))
        for name, value in self.data.items():
            print("Name: {}\nValue: {}\nKeys: {}".format(name, value, keys))
            temp = value.update(keys + '_' + name)
            arr.append(temp)
        return self.data

    def search_dict(self, subdict, arr=[]):
        for name, value in subdict.items():
            if type(value) is dict:
                arr = self.search_dict(subdict[name], arr)
            else:
                try:
                    arr.extend(value.get_value())
                except TypeError:
                    arr.extend([value.get_value()])
        return arr

    def get_value(self):
        return self.search_dict(self.data, [])




if __name__ == '__main__':
    """
    Possible keys:
    Position_F1/F2
    """
    pass



