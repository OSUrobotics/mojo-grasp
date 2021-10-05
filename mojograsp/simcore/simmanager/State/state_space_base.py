#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 14:57:24 2021
@author: orochi
"""

import json
import time
import numpy as np
from mojograsp.simcore.simmanager.State.State_Metric.state_metric import StateMetricAngle, StateMetricPosition, StateMetricGroup, StateMetricDistance
from collections import OrderedDict


class StateSpaceBase:
    valid_state_names = {'Angle': StateMetricAngle, 'Position': StateMetricPosition, 'StateGroup': StateMetricGroup,
                         'Distance': StateMetricDistance}
    _sim = None

    def __init__(self, path=None):
        with open(path) as f:
            self.json_data = json.load(f)
        self.data = OrderedDict()

    def get_obs(self):
        pass

    def update(self):
        pass


if __name__ == "__main__":
    try_state = StateSpaceBase()

    try_state.update()
    pass
