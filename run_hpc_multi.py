#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 12:35:23 2023

@author: orochi
"""

import pathlib
import sys
print('path',sys.path)
print('version',sys.version)

from demos.rl_demo import multiprocess_gym_run
#print('path',sys.path)
#print('version',sys.version)

def main(run_id):
    print(run_id)
    run_path_names = ['Full_continue','Rotation_continue']
    folder_names = ['JA_S3_larger_space_new_weight_contact']
    overall_path = pathlib.Path(__file__).parent.resolve()
    run_path = overall_path.joinpath('demos/rl_demo/data/')
    final_path = run_path.joinpath(run_path_names[run_id-1])
    final_path = final_path.joinpath(folder_names[0])
    print(str(final_path))
    multiprocess_gym_run.main(str(final_path) + '/experiment_config.json')
    
if __name__ == '__main__':
    # print(sys.argv)
    main(int(sys.argv[1]),num_cpu=int(sys.argv[2])
