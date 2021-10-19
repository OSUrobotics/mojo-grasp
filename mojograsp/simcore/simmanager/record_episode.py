#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 14:16:32 2021

@author: orochi
"""
from collections import OrderedDict
from mojograsp.simcore.simmanager.record_episode_base import RecordEpisodeBase


class RecordEpisode(RecordEpisodeBase):

    def __init__(self, identifier: str):
        """ Episode class contains the timesteps from a single episode, methods
        to acces the data and a method to save it in a csv file
        @param identifier - str containing name of episode used for identifying
        datafiles in the futureshape being picked up"""
        super().__init__(identifier)
        self.data = OrderedDict()
        self.identifier = identifier
        self.episode_number = 0

    @staticmethod
    def build_identifier(id_1):
        """ Static method to take in an arbitrary number of features and use
        them to build a unique identifier for the episode. Example features
        include episode number, shape being grabbed, hand used etc.
        @return identifier - str containing name of episode used for
        identifying datafiles in the futureshape being picked up"""
        return str(id_1)

    def add_timestep(self, timestep):
        """Method to add a new timestep to the episode
        @param timestep - RecordTimestep class containing all relevant data"""
        self.data['t_' + str(timestep.times['timestep'])] = timestep

    def get_timestep_as_dict(self, timestep):
        """Method to return the data in the timestep associated with the string
        or int given as an ordered dictionary
        @param timestep - int or string of desired timestep"""
        try:
            data = self.data[timestep]
        except KeyError:
            key = 't_'+str(timestep)
            data = self.data[key]
        return data.get_full_timestep()

    def get_full_episode_as_dict(self):
        """Method to return the data in all the timesteps in an ordered dict"""
        data = OrderedDict()
        for k, v in self.data.items():
            data[k] = v.get_full_timestep()
        return data

    def save_episode_as_csv(self, file_name=None, episode_number=0):
        """Method to save the episode
        @param file_name - name of file"""
        flag = True
        if file_name is None:
            file_name = self.identifier
        for i in self.data.values():
            if flag:
                i.save_timestep_as_csv(file_name=file_name+str(episode_number), write_flag='w', episode_number=episode_number)
                i.save_timestep_as_csv(file_name=file_name+"_all", write_flag='a', write_all=True, episode_number=episode_number)
                self.episode_number+=1
                flag = False
            else:
                i.save_timestep_as_csv(file_name=file_name+str(episode_number), write_flag='a')
                i.save_timestep_as_csv(file_name=file_name+"_all", write_flag='a', episode_number=episode_number)

    def save_episode_as_json(self, file_name=None):
        """Method to save the episode
        @param file_name - name of file"""
        if file_name is None:
            file_name = self.identifier
        for timestep_name, value in self.data.items():
            value.save_timestep_as_json(file_name=file_name + timestep_name)

