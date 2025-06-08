#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains HRL state designed to function on two different timescales, the manager scale and the worker scale
"""

from mojograsp.simcore.state import StateDefault
from demos.rl_demo.multiprocess_state import *
import numpy as np
from copy import deepcopy
from mojograsp.simobjects.two_finger_gripper import TwoFingerGripper
from mojograsp.simcore.goal_holder import *

class HRLState(MultiprocessState):
    pass