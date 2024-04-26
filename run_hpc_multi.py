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
    folder_names = ['FTP_S1','FTP_S2','FTP_S3','JA_S1','JA_S2','JA_S3']
    overall_path = pathlib.Path(__file__).parent.resolve()
    run_path = overall_path.joinpath('demos/rl_demo/data/HPC_slide_round2')
    final_path = run_path.joinpath(folder_names[run_id-1])
    print(str(final_path))
    multiprocess_gym_run.main(str(final_path) + '/experiment_config.json')
    
if __name__ == '__main__':
    # print(sys.argv)
    main(int(sys.argv[1]))
