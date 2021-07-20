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
from mojograsp.simcore.simmanager.State.State_Metric.state_metric_base import StateMetricBase


class StateMetricPyBullet(StateMetricBase):

    def get_value(self):
        return self.data.value


class StateMetricAngle(StateMetricPyBullet):
    pass


class StateMetricPosition(StateMetricPyBullet):
    pass


class StateMetricVector(StateMetricPyBullet):
    pass


class StateMetricRatio(StateMetricPyBullet):
    pass


class StateMetricDistance(StateMetricPyBullet):
    pass


class StateMetricDotProduct(StateMetricPyBullet):
    pass


if __name__ == '__main__':
    pass
