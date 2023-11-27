#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on June 29 2023

@author: orochi
"""
import pathlib
import sys
from mojograsp.simcore import gym_run

def main(run_id):
    #print(run_id)
    folder_names = ['full_big']
    
    overall_path = pathlib.Path(__file__).parent.resolve()
    run_path = overall_path.joinpath('demos/rl_demo/data/')
    final_path = run_path.joinpath(folder_names[run_id-1])
    print(str(final_path))
    gym_run.run_pybullet(str(final_path) + '/experiment_config.json',runtype='run')
    
if __name__ == '__main__':
    # print(sys.argv)
    main(int(sys.argv[1]))
