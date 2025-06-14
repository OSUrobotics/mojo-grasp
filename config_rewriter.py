#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 13:55:03 2023

@author: orochi
"""
import json
import os
import pathlib

folder_path = './demos/rl_demo/data/Static_1/'


overall_path = pathlib.Path(__file__).parent.resolve()
resource_path = overall_path.joinpath('demos/rl_demo/resources')
run_path = overall_path.joinpath('demos/rl_demo/runs')
batch_run_folder = overall_path.joinpath(folder_path)

subfolders = os.listdir(batch_run_folder)

if 'experiment_config.json' in subfolders:
    print('Folder path is a single configuration')

    curr_folder = str(batch_run_folder)
    with open(curr_folder+'/experiment_config.json', 'r') as file:
        config = json.load(file)
    high_level_path = config['save_path'].split('/demos/')[0]
    print('hlp', high_level_path)
    print('overall folder', str(overall_path))
    print('overall path', )
    for k in config.keys():
        if k == 'save_path':
            print('old',k,config[k])
            config[k] = config[k].replace(config[k], str(batch_run_folder)+'/')
            print('new',k,config[k])
        elif type(config[k]) == str:
            print('old',k,config[k])
            config[k] = config[k].replace(high_level_path, str(overall_path))
            print('new',k,config[k])
        elif type(config[k]) == list:
            for i in range(len(config[k])):
                if type(config[k][i])==str:
                    print('old',k,config[k][i])
                    config[k][i] = config[k][i].replace(high_level_path, str(overall_path))
                    print('new',k,config[k][i])
    with open(curr_folder+'/experiment_config.json', 'w') as file:
        json.dump(config,file)
else:
    print('Folder path is a folder with subfolders')
    for folder in subfolders:
        curr_folder = str(batch_run_folder.joinpath(folder))
        with open(curr_folder+'/experiment_config.json', 'r') as file:
            config = json.load(file)
        high_level_path = config['save_path'].split('/demos/')[0]
        print('hlp', high_level_path)
        print('overall folder', str(overall_path))
        for k in config.keys():
            if k == 'save_path':
                print('old',k,config[k])
                config[k] = config[k].replace(config[k], curr_folder + '/')
                print('new',k,config[k])
            if type(config[k]) == str:
                print('old',k,config[k])
                config[k] = config[k].replace(high_level_path, str(overall_path))
                print('new',k,config[k])
            elif type(config[k]) == list:
                for i in range(len(config[k])):
                    if type(config[k][i])==str:
                        print('old',k,config[k][i])
                        config[k][i] = config[k][i].replace(high_level_path, str(overall_path))
                        print('new',k,config[k][i])

        with open(curr_folder+'/experiment_config.json', 'w') as file:
            json.dump(config,file)
