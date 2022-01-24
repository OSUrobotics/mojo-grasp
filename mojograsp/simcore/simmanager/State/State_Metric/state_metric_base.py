#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 13:59:44 2021
@author: orochi
"""

import os
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mojograsp.simcore.datacollection.stats_tracker_base import *


class StateMetricBase:
    _sim = None

    def __init__(self, data_structure):
        # Remember that you are using an inherited class of StatsTrackerBase to Scale data between min and max
        print('intializing with data strucure:', data_structure)
        try:
            self.data = StatsTrackerArrayScaled(data_structure[0], data_structure[1])
            print('stats tracker array initialized with min and max',
                  self.data.allowable_min, self.data.allowable_max)
        except TypeError:
            self.data = StatsTrackerBaseScaled(data_structure[0], data_structure[1])
            print('stats tracker base initialized with min and max',
                  self.data.allowable_min, self.data.allowable_max)
        except KeyError:
            self.data = []

    def get_value(self):
        return self.data.value

    def update(self, keys):
        pass

    def norm_data(self, value):
        return list(value) # /np.linalg.norm(value)
