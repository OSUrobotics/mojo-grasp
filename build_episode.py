#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 08:53:49 2023

@author: orochi
"""



import os
import re
import pickle as pkl

folder = '/home/orochi/Downloads/'
episode_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith('.pkl')]
filenames_only = [f for f in os.listdir(folder) if f.lower().endswith('.pkl')]
real_ones = [f for f in filenames_only if f[0].isupper()]
print(filenames_only)
print(real_ones)
for filename in real_ones:
    direction = re.findall('\w+_',filename)
    print(direction[0])
    # a = [f for f in filenames_only if f.startswith(direction[0])]
    with open(folder+direction[0]+'state.pkl', 'rb') as file:
        state = pkl.load(file)
    with open(folder+direction[0]+'actor.pkl', 'rb') as file:
        action = pkl.load(file)
    edict = {'number': 0, 'timestep_list':[]}
    for i,j in zip(state,action):
        edict['timestep_list'].append({'state':i,'action':j})
    with open(folder+direction[0] +'episode.pkl','wb') as file:
        pkl.dump(edict,file)