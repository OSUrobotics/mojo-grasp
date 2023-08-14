#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 12:45:29 2023

@author: orochi
"""
from mojograsp.simcore.gym_run import run_pybullet

action = [-.5,-.9,-1,0]
action1 = [-1,-0,-1,-1]
action2 = [action.copy() for _ in range(50)]
action3 = [action1.copy() for _ in range(100)]
action2.extend(action3)
print(action2)
# run_pybullet('/home/orochi/mojo/mojo-grasp/demos/rl_demo/data/ftp_long_rand_end/experiment_config.json',runtype='eval')
run_pybullet('/home/mothra/mojo-grasp/demos/rl_demo/data/test_throwaway/experiment_config.json',runtype='replay', action_list=action2)