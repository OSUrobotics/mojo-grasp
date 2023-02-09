#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 11:01:50 2023

@author: orochi
"""

import pickle as pkl
import os
import re
import numpy as np

class data_processor():
    def __init__(self, filepath):
       self.data_path = filepath 
       self.episode_data = []
       self.save_all_flag = True
       
    def load_data(self):
        """
        Method called after all episodes are completed. Loads all episodes in
        a folder into the class to be combined and saved later
        """
        all_names = os.listdir(self.data_path)
        pkl_names = []
        pkl_nums = []
        for name in all_names:
            # print(name)
            if "all" in name and ".pkl" in name:
                print('Folder already has episode all. Data not loaded')
                self.episode_data = []
                self.save_all_flag = False
            elif '.pkl' in name:
                pkl_names.append(name)
                # print(name)
                temp = re.search('\d+',name)
                pkl_nums.append(int(temp[0]))
        pkl_sort = np.argsort(pkl_nums)
        new_pkl_names = []
        for ind in pkl_sort:
            new_pkl_names.append(pkl_names[ind])
        for name in new_pkl_names:
            with open(self.data_path + name, 'rb') as datafile: 
                self.episode_data.append(pkl.load(datafile))
        # print(data_list.keys())
        
    def save_all(self):
        """
        Method called after all episodes are completed. Saves all
        episode dictionaries to a pkl file. 
        """
        if self.save_all_flag and self.data_path != None:
            file_path = self.data_path + \
                "episode_all.pkl"
            with open(file_path, 'wb') as fout:
                self.episodes = {"episode_list": self.episode_data}
                pkl.dump(self.episodes, fout)