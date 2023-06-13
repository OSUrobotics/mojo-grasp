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
    def __init__(self, filepath, eval_flag=False):
       self.data_path = filepath 
       self.episode_data = []
       self.save_all_flag = True
       self.eval_flag = eval_flag
       
    def load_data(self):
        """
        Method called after all episodes are completed. Loads all episodes in
        a folder into the class to be combined and saved later
        """
        all_names = os.listdir(self.data_path)
        pkl_names = []
        pkl_nums = []
        for name in all_names:
            if "all" in name and ".pkl" in name:
                if not (('Evaluation' in name) ^ self.eval_flag):
                    print('Folder already has episode all. Data not loaded')
                    print(name)
                    self.episode_data = []
                    self.save_all_flag = False
                    return
            elif '.pkl' in name and 'sampled' not in name:
                # print('first stage')
                if not (('Evaluation' in name) ^ self.eval_flag):
                    print(name)
                    pkl_names.append(name)
                    temp = re.search('\d+',name)
                    pkl_nums.append(int(temp[0]))
        pkl_sort = np.argsort(pkl_nums)
        new_pkl_names = []
        print('going to next')
        for ind in pkl_sort:
            new_pkl_names.append(pkl_names[ind])
        print('found names: ', len(new_pkl_names))
        for name in new_pkl_names:
            with open(self.data_path + name, 'rb') as datafile: 
                self.episode_data.append(pkl.load(datafile))
        self.save_all_flag = True
        
    def save_all(self):
        """
        Method called after all episodes are completed. Saves all
        episode dictionaries to a pkl file. 
        """
        if self.save_all_flag and self.data_path != None:
            if not self.eval_flag:
                file_path = self.data_path + \
                    "episode_all.pkl"
                with open(file_path, 'wb') as fout:
                    self.episodes = {"episode_list": self.episode_data}
                    pkl.dump(self.episodes, fout)
            else:
                file_path = self.data_path + \
                    "Evaluation_episode_all.pkl"
                with open(file_path, 'wb') as fout:
                    self.episodes = {"episode_list": self.episode_data}
                    pkl.dump(self.episodes, fout)
        print('save completed')
     

def main():
    path ='/home/orochi/mojo/mojo-grasp/demos/rl_demo/data/0_slope_epsilon_greedy_all_noises/'
    d = data_processor(path + 'Train/')
    d.load_data()
    d.save_all()
    d = data_processor(path + 'Test/', eval_flag=True)
    d.load_data()
    d.save_all()
        
if __name__ == '__main__':
    main()