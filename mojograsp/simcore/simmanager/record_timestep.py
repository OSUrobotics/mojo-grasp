#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 13:06:46 2021

@author: orochi
"""
import os
import time
import csv
from copy import deepcopy
from collections import OrderedDict
import json
#from mojograsp.simcore.simmanager.record_timestep_base import RecordTimestepBase


class RecordTimestep():
    _sim = None

    def __init__(self, phase, data_path=None):
        """ Timestep class contains the state, action, reward and time of a
        moment in the simulator. It contains this data as their respective
        classes but has methods to return the data contained in them and save
        the data to a csv
        @param phase - Phase class containing the state, action, reward as
        State, Action and Reward classes and the timestep and sim time as int
        and float"""
        self.state = deepcopy(phase.state)
        # self.action = deepcopy(phase.Action)
        self.action = phase.Action
        self.reward = deepcopy(phase.reward)
        self.times = {'wall_time': time.time(), 'sim_time': deepcopy(self._sim.curr_simstep),
                      'timestep': deepcopy(self._sim.curr_timestep)}
        self.phase = phase.name
        self.data_path = data_path+'/timesteps/'
        self.data = OrderedDict()
        self.get_full_timestep()

    def get_state_as_arr(self):
        """Method to return the state data as a single list
        @return - list of state values"""
        return self.state.get_obs()

    def get_action_as_arr(self):
        """Method to return the action data as a single list
        @return - list of action values"""
        action_profile = self.action.get_action()
        return list(action_profile[-1])

    def get_reward_as_arr(self):
        """Method to return the reward data as a single list
        @return - list of reward values"""
        try:
            reward, _ = self.reward.get_reward()
        except AttributeError:
            reward = None
        return [reward]

    def get_full_timestep(self):
        """Method to return all stored data as one dictionary of lists"""

        self.data['phase'] = self.phase
        self.data['state'] = self.get_state_as_arr()
        self.data['action'] = self.get_action_as_arr()
        self.data['reward'] = self.get_reward_as_arr()
        self.data.update(self.times)
        return self.data

    def save_timestep_as_csv(self, file_name=None, header=False, episode_number=0):
        """Method to save the timestep in a csv file
        @param file_name - name of file
        @param write_flag - type of writing, defaults to write but can be set
        to 'a' to append to the file instead"""
        if file_name is None:
            file_name = self.data_path + 'tstep_' + str(self.times['timestep']) +\
            '_wall_time_' + time.strftime("%m-%d-%y %H:%M:%S", time.localtime(self.times['wall_time'])) + '.csv'

        # data = self.get_full_timestep()
        header_text = ['Episode:'+str(episode_number),'Phase', 'WallTime', 'SimTime', 'TimeStep', 
                  'State:'+str(len(self.data['state'])), 'Action:'+str(len(self.data['action'])),
                  'Reward:'+str(len(self.data['reward']))]

        with open(file_name, 'a', newline='') as csv_file:
            time_writer = csv.writer(csv_file, delimiter=',', quotechar='|',
                                     quoting=csv.QUOTE_MINIMAL)
            if header:
                time_writer.writerow(header_text)
            time_writer.writerow([self.data['phase']] + [self.data['wall_time']] + [self.data['sim_time']] +
                                 [self.data['timestep']] + self.data['state'] +
                                 self.data['action'] + self.data['reward'])


    def save_timestep_as_json(self, file_name=None):
        """Method to save the timestep in a json file
        @param file_name - name of file"""
        path = os.path.dirname(os.path.abspath(__file__))
        if file_name is None:
            file_name = path + '/data/' + 'tstep_' + str(self.times['timestep']) +\
            '_wall_time_' + time.strftime("%m-%d-%y %H:%M:%S", time.localtime(self.times['wall_time'])) + '.json'
        else:
            file_name = path + '/data/' + file_name + '.json'
        try:
            reward_data = self.reward.get_reward()
        except AttributeError:
            reward_data = None

        with open(file_name, 'w') as json_file:
            json.dump(self.times, json_file)
            json.dump(self.state.get_data_dict(), json_file)
            json.dump(self.action.get_action_dict(), json_file)
            json.dump(reward_data, json_file)

# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Thu Sep 23 13:06:46 2021
#
# @author: orochi
# """
# import os
# import time
# import csv
# from copy import deepcopy
# from collections import OrderedDict
# import json
# #from mojograsp.simcore.simmanager.record_timestep_base import RecordTimestepBase
#
#
# class RecordTimestep():
#     _sim = None
#
#     def __init__(self, phase, data_path=None):
#         """ Timestep class contains the state, action, reward and time of a
#         moment in the simulator. It contains this data as their respective
#         classes but has methods to return the data contained in them and save
#         the data to a csv
#         @param phase - Phase class containing the state, action, reward as
#         State, Action and Reward classes and the timestep and sim time as int
#         and float"""
#         self.state = deepcopy(phase.state)
#         # self.action = deepcopy(phase.Action)
#         self.action = phase.Action
#         self.reward = deepcopy(phase.reward)
#         self.times = {'wall_time': time.time(), 'sim_time': deepcopy(self._sim.curr_simstep),
#                       'timestep': deepcopy(self._sim.curr_timestep)}
#         self.phase = phase.name
#         self.data_path = data_path+'/timesteps/'
#
#     def get_state_as_arr(self):
#         """Method to return the state data as a single list
#         @return - list of state values"""
#         return self.state.get_obs()
#
#     def get_action_as_arr(self):
#         """Method to return the action data as a single list
#         @return - list of action values"""
#         action_profile = self.action.get_action()
#         return list(action_profile[-1])
#
#     def get_reward_as_arr(self):
#         """Method to return the reward data as a single list
#         @return - list of reward values"""
#         try:
#             reward, _ = self.reward.get_reward()
#         except AttributeError:
#             reward = None
#         return [reward]
#
#     def get_full_timestep(self):
#         """Method to return all stored data as one dictionary of lists"""
#         data = OrderedDict()
#         data['phase'] = self.phase
#         data['state'] = self.get_state_as_arr()
#         data['action'] = self.get_action_as_arr()
#         data['reward'] = self.get_reward_as_arr()
#         data.update(self.times)
#         return data
#
#     def save_timestep_as_csv(self, file_name=None, header=False, episode_number=0):
#         """Method to save the timestep in a csv file
#         @param file_name - name of file
#         @param write_flag - type of writing, defaults to write but can be set
#         to 'a' to append to the file instead"""
#         if file_name is None:
#             file_name = self.data_path + 'tstep_' + str(self.times['timestep']) +\
#             '_wall_time_' + time.strftime("%m-%d-%y %H:%M:%S", time.localtime(self.times['wall_time'])) + '.csv'
#
#         data = self.get_full_timestep()
#         header_text = ['Episode:'+str(episode_number),'Phase', 'WallTime', 'SimTime', 'TimeStep',
#                   'State:'+str(len(data['state'])), 'Action:'+str(len(data['action'])),
#                   'Reward:'+str(len(data['reward']))]
#
#         with open(file_name, 'a', newline='') as csv_file:
#             time_writer = csv.writer(csv_file, delimiter=',', quotechar='|',
#                                      quoting=csv.QUOTE_MINIMAL)
#             if header:
#                 time_writer.writerow(header_text)
#             time_writer.writerow([data['phase']] + [data['wall_time']] + [data['sim_time']] +
#                                  [data['timestep']] + data['state'] +
#                                  data['action'] + data['reward'])
#
#
#     def save_timestep_as_json(self, file_name=None):
#         """Method to save the timestep in a json file
#         @param file_name - name of file"""
#         path = os.path.dirname(os.path.abspath(__file__))
#         if file_name is None:
#             file_name = path + '/data/' + 'tstep_' + str(self.times['timestep']) +\
#             '_wall_time_' + time.strftime("%m-%d-%y %H:%M:%S", time.localtime(self.times['wall_time'])) + '.json'
#         else:
#             file_name = path + '/data/' + file_name + '.json'
#         try:
#             reward_data = self.reward.get_reward()
#         except AttributeError:
#             reward_data = None
#
#         with open(file_name, 'w') as json_file:
#             json.dump(self.times, json_file)
#             json.dump(self.state.get_data_dict(), json_file)
#             json.dump(self.action.get_action_dict(), json_file)
#             json.dump(reward_data, json_file)