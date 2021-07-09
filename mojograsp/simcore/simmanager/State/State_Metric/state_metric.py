#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 14:57:24 2021
@author: orochi
"""
import time
import numpy as np
import re
from pathlib import Path
import os
import sys
import pybullet as p
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mojograsp.simcore.datacollection.stats_tracker_base import *
from collections import OrderedDict


class StateMetric:
    _sim = None

    def __init__(self, data_structure, hand, obj):
        self.hand = hand
        self.object = obj
        flag = True
        print('intializing with data strucure:', data_structure)
        try:
            self.data = StatsTrackerArray(data_structure[0], data_structure[1])
            print('stats tracker array initialized with min and max', self.data.allowable_min, self.data.allowable_max)
        except TypeError:
            self.data = StatsTrackerBase(data_structure[0], data_structure[1])
            print('stats tracker base initialized with min and max', self.data.allowable_min, self.data.allowable_max)
        except KeyError:
            self.data = []

    def get_value(self):
        return self.data.value

    @staticmethod
    def get_geom_name(keys):
        """
        TODO: Create a mapping between JSON nams and joint indices, etc
        :param keys:
        :return:
        """
        print("KEYS: {}".format(keys))

        return

    def update(self):
        print("Updating:".format())
        return
