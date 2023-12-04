#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 12:35:23 2023

@author: orochi
"""

import pathlib
import sys
from demos.rl_demo import multiprocess_gym_run

def main(run_id):
    #print(run_id)
    folder_names = ['FTP_halfstate_A_rand','JA_halfstate_A_rand']
    
    overall_path = pathlib.Path(__file__).parent.resolve()
    run_path = overall_path.joinpath('demos/rl_demo/data/')
    final_path = run_path.joinpath(folder_names[run_id-1])
    print(str(final_path))
    multiprocess_gym_run.main(str(final_path) + '/experiment_config.json')
    
if __name__ == '__main__':
    # print(sys.argv)
    main(int(sys.argv[1]))
