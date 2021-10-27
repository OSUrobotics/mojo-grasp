#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 14:57:24 2021
@author: orochi
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import OrderedDict
from mojograsp.simcore.simmanager.State.State_Metric.state_metric_base import StateMetricBase


class StateMetricPyBullet(StateMetricBase):

    def get_value(self):
        return self.data.value

    def get_index_from_keys(self, keys):
        if 'F1_l' in keys:
            if 'Prox' in keys:
                index_name = 'l_prox_pin'
            elif 'Dist' in keys:
                index_name = 'l_distal_pin'
            else:
                print("Wrong Key!")
                raise KeyError

        elif 'F2_r' in keys:
            if 'Prox' in keys:
                index_name = 'l_prox_pin'
            elif 'Dist' in keys:
                index_name = 'l_distal_pin'
            else:
                print("Wrong Key!")
                raise KeyError

        elif 'Palm' in keys:
            index = -1
            return index

        else:
            print("Wrong Key!")
            raise KeyError

        index = self._sim.hand.joint_index[index_name]

        return index


class StateMetricAngle(StateMetricPyBullet):
    def update(self, keys):
        joint_index = self.get_index_from_keys(keys)
        curr_joint_angles = StateMetricBase._sim.get_hand_curr_joint_angles([joint_index])
        # print("CURR JOINT ANGLES: {}".format(curr_joint_angles))
        self.data.set_value(curr_joint_angles)


class StateMetricPosition(StateMetricPyBullet):
    def update(self, keys):
        """
        TODO: Change numbers to keys. use keys to derive joint indices
        :param keys:
        :return:
        """
        if "Obj" in keys:
            # if "in_Start" in keys:
            #     obj_full_pose = StateMetricBase._sim.get_obj_curr_pose_in_start_pose()
            # elif "Target" in keys:
            #     obj_full_pose = StateMetricBase._sim.get_obj_target_pose()
            #     print("@@@@@TARGET POSE:", obj_full_pose)
            # else:
            #     obj_full_pose = StateMetricBase._sim.get_obj_curr_pose()
            obj_full_pose = StateMetricBase._sim.get_obj_curr_pose()

            curr_pose = [obj_full_pose[0][0], obj_full_pose[0][1], obj_full_pose[0][2], obj_full_pose[1][0],
                         obj_full_pose[1][1], obj_full_pose[1][2], obj_full_pose[1][3]]

        else:
            joint_index = self.get_index_from_keys(keys)
            if joint_index == -1:
                curr_pose = StateMetricBase._sim.get_hand_curr_pose()[0]
            else:
                curr_pose = StateMetricBase._sim.get_curr_link_pos([joint_index])
        self.data.set_value(curr_pose)


class StateMetricDistance(StateMetricPyBullet):
    def update(self, keys):
        if 'ObjSize' in keys:
            dimensions = StateMetricBase._sim.get_obj_dimensions()
            self.data.set_value(dimensions)

        elif 'FingerObj' in keys:
            joint_index = self.get_index_from_keys(keys)
            contact_info = self.get_contact_info(joint_index)
            try:
                finger_obj_distance = contact_info[0][6]
            except IndexError:
                finger_obj_distance = self.data.allowable_max
            self.data.set_value(finger_obj_distance)

    def get_contact_info(self, joint_index_num):
        contact_points_info = StateMetricBase._sim.get_contact_info(joint_index_num)
        return contact_points_info


class StateMetricGroup(StateMetricPyBullet):
    valid_state_names = {'Position': StateMetricPosition, 'Distance': StateMetricDistance, 'Angle': StateMetricAngle,
                         'StateGroup': 'StateMetricGroup'}

    def __init__(self, data_structure):
        super().__init__(data_structure)
        self.data = OrderedDict()
        for name, value in data_structure.items():
            state_name = name.split('_')
            try:
                self.data[name] = StateMetricGroup.valid_state_names[state_name[0]](value)
            except TypeError:
                self.data[name] = StateMetricGroup(value)
            except KeyError:
                print('Invalid state name. Valid state names are', [name for name in
                                                                    StateMetricGroup.valid_state_names.keys()])

    def update(self, keys):
        arr = []
        # print("Here Metric Group: {}\nKeys: {}".format(self.data, keys))
        for name, value in self.data.items():
            # print("Name: {}\nValue: {}\nKeys: {}".format(name, value, keys))
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



