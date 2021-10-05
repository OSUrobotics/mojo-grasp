#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 11:56:31 2021
@author: orochi
"""
import json
import numpy as np
import os
import sys
import warnings
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mojograsp.simcore.datacollection.stats_tracker_base import StatsTrackerArray
from collections import OrderedDict


class Action:
    _sim = None

    def __init__(self, starting_action_units=None, differentiated_action_units_range=[0.2, 20],
                 json_path='/Users/asar/Desktop/Grimm\'s Lab/Manipulation/PyBulletStuff/mojo-grasp/mojograsp/simcore/simmanager/Action/action.json'):
        """
        Starting action_units is the action_units of the fingers at initialization. We
        assume the initial action_units is 0 if not otherwise specified.
        @param desired_action_units - list of action_units with length described in
        action.json
        @param json_path - path to json file
        """
        if differentiated_action_units_range is None:
            differentiated_action_units_range = [0.2, 20]
        self.min_differentiated_action_units = differentiated_action_units_range[0]  # rad/s^2
        self.max_differentiated_action_units = differentiated_action_units_range[1]  # rad/s^2
        self.action_profile = []

        with open(json_path) as f:
            json_conts = json.load(f)

        # Pull the important parameters from the json file
        # They will be changed to class varriables when we add a function to
        # autogenerate a json file with these parameters and others when the
        # simulator starts.

        if self._sim is not None:
            self.time = self._sim.sim_sleep # length of a timestep in seconds
            self.timesteps = self._sim.sim_step # number of simulation
        else:
            params = json_conts['Parameters']
            self.time = params['Timestep_len']
            self.timesteps = params['Timestep_num']

        # Set up the current and last action_units values with max and min action_unitss and
        # the order of actions pulled from the json file
        action_struct = json_conts['Action']
        self.action_order = action_struct.keys()
        # print("##@@## ACTION ORDER: {}".format(self.action_order))
        action_size_list = []
        for key in self.action_order:
            self.link_order = json_conts['Action'][key]
            # print("##@@## LINK ORDER: {}".format(self.link_order))

            for value in self.link_order.values():
                action_size_list.append(list(value))
                # print("ACTION SIZE LIST: {}".format(action_size_list))
        action_min_and_max = np.array(action_size_list)
        # print("ACTION MIN MAX: {} {}".format(action_min_and_max, action_min_and_max[:,0]))
        self.current_action_units = StatsTrackerArray(action_min_and_max[:, 0],
                                                   action_min_and_max[:, 1])
        self.last_action_units = StatsTrackerArray(action_min_and_max[:, 0],
                                                action_min_and_max[:, 1])
        # print("CURR action_units: {}\nLAST action_units: {}".format(self.current_action_units, self.last_action_units))

        # set initial values of the action_units
        try:
            self.last_action_units.set_value(starting_action_units)
            self.current_action_units.set_value(starting_action_units)
        except TypeError:
            self.last_action_units.set_value(np.zeros(len(action_min_and_max)))
            self.current_action_units.set_value(np.zeros(len(action_min_and_max)))

    def get_action(self):
        """Returns the action_unitss to get from old action_units to new action_units as a list
        of lists"""
        return self.action_profile

    def build_action(self):
        """Builds the action profile (action_units profile to get from old action_units to
        new action_units)"""
        action_units = np.array(self.last_action_units.value)
        ending_action_units = np.array(self.current_action_units.value)
        direction = [np.sign(ending_action_units[i]-action_units[i]) for i in
                     range(len(action_units))]
        action_profile = np.zeros([self.timesteps, len(action_units)])
        if any(abs(ending_action_units - action_units) / (self.time*self.timesteps) >
               self.max_differentiated_action_units):
            warnings.warn('Desired action_units is too different from current\
                  action_units to reach in ' + str(self.timesteps) + ' timesteps.\
                  Action will apply max differentiated_action_units for all steps but \
                  this will not reach the desired action_units!')
        for i in range(len(action_units)):
            if direction[i] > 0:
                action_profile[0][i] = min(action_units[i]+self.max_differentiated_action_units
                                           * self.time, ending_action_units[i])
            else:
                action_profile[0][i] = max(action_units[i]-self.max_differentiated_action_units
                                           * self.time, ending_action_units[i])
        for j in range(self.timesteps-1):
            for i in range(len(action_units)):
                if direction[i] > 0:
                    action_profile[j+1][i] = min(action_profile[j][i]
                                                 + self.max_differentiated_action_units *
                                                 self.time, ending_action_units[i])
                else:
                    action_profile[j+1][i] = max(action_profile[j][i]
                                                 - self.max_differentiated_action_units *
                                                 self.time, ending_action_units[i])
        return action_profile

    def set_action_units(self, action_units):
        """sets last action_units to current action_units's value and sets current action_units to
        input action_units value, then calculates the new action profile"""
        if len(action_units) != len(self.last_action_units.value):
            raise IndexError('desired action_units is not the same length as current action_units, action_units not set. action_units should have '
                             'length '+ str(len(self.last_action_units.value)) + ' but has length {}'.format(len(action_units)))
        self.last_action_units.set_value(self.current_action_units.value)
        self.current_action_units.set_value(action_units)
        self.action_profile = self.build_action()
        return self.action_profile

    def get_name_value(self):
        """sets action_units action_units and calculates the new action profile"""
        action_dict = OrderedDict()
        for i, j in zip(self.action_order, self.current_action_units.value):
            action_dict[i] = j
        return action_dict


if __name__ == "__main__":
    a = Action()
    print(a.set_action_units([0, 0, 0, 0.3, 0.3, 0.3]))
    print(a.set_action_units([0.2, 0.5, 0.1, 0.5, 0.9, 0.0]))
