#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 15:26:01 2024

@author: orochi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

theta = np.random.uniform(0,np.pi*2,1200)

x_wall = np.sin(theta) * 0.08
y_wall = np.cos(theta) * 0.08

g_theta = np.random.uniform(0.62,1.57,1200)
s_theta = np.random.uniform(0.945,1.57,1200)

s_r, g_r = [],[]
for s,g in zip(s_theta, g_theta):
    stemp = np.random.uniform(4.05/np.sin(s),5)
    gtemp = np.random.uniform(4.05/np.sin(g),7)
    s_r.append(stemp)
    g_r.append(gtemp)
    assert stemp*np.sin(s) > 4.05
    assert gtemp*np.sin(g) > 4.05
    assert stemp*np.cos(s) > -0.1
    assert gtemp*np.cos(s) > -0.1

R_or_L = np.random.randint(0,2,1200)
start_x = []
start_y = []
goal_x = []
goal_y = []
for t,s,g,b,sr,gr in zip(theta,s_theta,g_theta,R_or_L,s_r,g_r):
    if b:
        start_x.append(np.sin(t-s)*sr)
        start_y.append(np.cos(t-s)*sr)
        goal_x.append(np.sin(t+g)*gr)
        goal_y.append(np.cos(t+g)*gr)
    else:
        start_x.append(np.sin(t+s)*sr)
        start_y.append(np.cos(t+s)*sr)
        goal_x.append(np.sin(t-g)*gr)
        goal_y.append(np.cos(t-g)*gr)

start_x = np.array(start_x)
start_y = np.array(start_y)
goal_x = np.array(goal_x)
goal_y = np.array(goal_y)
s_theta = np.array(s_theta)
g_theta = np.array(g_theta)
plt.scatter(x_wall,y_wall)
plt.show()

# theta = (theta-np.pi/2)%(np.pi*2)

df = pd.DataFrame(
    {'x_wall': x_wall,
      'y_wall': y_wall,
      'ang_wall':theta,
      'x_start': start_x,
      'y_start': start_y,
      'goal_x': goal_x,
      'goal_y': goal_y
      })
df.to_csv("resources/test_wall_poses.csv", index=False)

