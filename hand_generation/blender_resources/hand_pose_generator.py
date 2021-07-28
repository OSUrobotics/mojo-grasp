#!/usr/bin/env python3

import os
import sys
import bpy
import csv
import json
import glob
from pathlib import Path
from numpy import sin, cos, pi
import math


class hand_pose_generator():

    def __init__(self, hand_model):
        


        self.finger_width = 
        self.palm_width =  
        self.joint_length = 
        self.palm_height =

        self.finger_length_l = 
        

        
        self.finger_length_r = 


    def precision_close(self):
        pass

    def precision_open(self):
        pass

    def power_close(self):
        pass
        
    def power_open(self):
        pass
    


if __name__ == '__main__':

    hand_model = {
        "file_loc": "/hand_models/testing1/obj_files/",
        "palm": {
            "file_name": "body_palm_lpinvrspring",
            "palm_width": 0.14,
            "palm_height": 0.04,
            "palm_thickness": 0.02
        },
        "left_finger": {
            "file_names": [""],
            "segment_lengths": [],
            "finger_width": ,
            "joint_lengths": []  
        },
        "right_finger":{
            "file_names": [""],
            "segment_lengths": [],
            "finger_width": ,
            "joint_lengths": []
        }
        
    }