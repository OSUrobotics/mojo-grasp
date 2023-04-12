#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 12:58:13 2023

@author: orochi
"""

import pybullet as p
import pybullet_data
from mojograsp.simcore.run_from_file import run_pybullet

#going to have to check this, could be useful for setting up a bunch of trials to run
folder_names = ['','','']
overall_name = ''

for name in folder_names:
    run_pybullet(overall_name + name)
    p.disconnect()