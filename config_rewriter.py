#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 13:55:03 2023

@author: orochi
"""
import json
import os
import pathlib

folder_path = './demos/rl_demo/data/hpc2'

overall_path = pathlib.Path(__file__).parent.resolve()
resource_path = overall_path.joinpath('demos/rl_demo/resources')
run_path = overall_path.joinpath('demos/rl_demo/runs')
batch_run_folder = overall_path.joinpath(folder_path)

subfolders = os.listdir(batch_run_folder)

for folder in subfolders:
    print("FOLDER", folder)
    curr_folder = str(batch_run_folder.joinpath(folder))
    with open(curr_folder+'/experiment_config.json', 'r') as file:
        config = json.load(file)
    high_level_path = pathlib.Path(config['save_path']).parent.resolve(
    ).parent.resolve().parent.resolve().parent.resolve().parent.resolve()
    high_level_path = str(high_level_path)
    print('hlp', high_level_path)
    print('overall folder', str(overall_path))
    for k in config.keys():
        if type(config[k]) == str:
            print('old', k, config[k])
            print("BEFORE", config[k], str(overall_path), high_level_path)
            config[k] = config[k].replace(high_level_path, str(overall_path))
            print("AFTER", config[k])
            print('new', k, config[k])
    with open(curr_folder+'/experiment_config.json', 'w') as file:
        json.dump(config, file)