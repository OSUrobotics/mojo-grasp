#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 12:19:41 2022

@author: orochi
"""
import PySimpleGUI as sg
import os
import pickle as pkl
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import json
from PIL import ImageGrab
from scipy.stats import kde
import re
import matplotlib.pyplot as plt




def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

class plotMaker():
    
    def __init__(self, path):
        all_names = os.listdir(path)
        eval_pkl_names = []
        for name in all_names:
            if "Evaluation" in name:
                if '.pkl' in name:
                    eval_pkl_names.append(name)
            else:
                if '.pkl' in name and 'all' in name:
                    train_pkl_name = name
        self.eval_data = []
        for name in eval_pkl_names:
            with open(path + name, 'rb') as datafile: 
                temp = pkl.load(datafile)
                self.eval_data.append({'number':-temp['number'],'timestep_list':temp['timestep_list'].copy()})
        inds = np.argsort([self.eval_data[i]['number'] for i in range(len(self.eval_data))])
        self.eval_data = [self.eval_data[i] for i in inds]
        with open(path + train_pkl_name, 'rb') as datafile: 
            self.test_data = pkl.load(datafile)['episode_list']
        
    
    def plot_f1f2(self):
        f1_max = []
        f2_max = []
        f1_avg = []
        f2_avg = []
        f1_std = []
        f2_std = []
        
        for episode in self.test_data:
            num_timesteps = len(episode['timestep_list'])
            f1s = [episode['timestep_list'][i]['state']['f1_obj_dist'] for i in range(num_timesteps)]
            f2s = [episode['timestep_list'][i]['state']['f2_obj_dist'] for i in range(num_timesteps)]
            f1_max.append(np.max(f1s))
            f2_max.append(np.max(f2s))
            f1_avg.append(np.average(f1s))
            f2_avg.append(np.average(f2s))
            f1_std.append(np.std(f1s))
            f2_std.append(np.std(f2s))
        num_episodes = len(self.test_data)
        plt.plot(range(num_episodes), f1_max)
        plt.plot(range(num_episodes), f1_max)
        plt.xlabel('Episode')
        plt.ylabel('Distance (m)')
        plt.title('F1 and F2 max object distances')
        plt.legend(['F1 obj distance', 'F2 obj distance'])
        plt.show()        
        plt.clf()
        plt.errorbar(range(num_episodes), f1_avg, f1_std)
        plt.errorbar(range(num_episodes), f2_avg, f2_std)
        plt.xlabel('Episode')
        plt.ylabel('Distance (m)')
        plt.title('F1 and F2 average object distances')
        plt.legend(['F1 obj distance', 'F2 obj distance'])
        plt.show()        

    def plot_sf_stuff(self,test=True):   
        s_or_f = []
        fails = []
        success_timesteps = []
        success_vel = []
        fail_timesteps = []
        dxdy_final = []
        dxdy_closest = []
        cut_short = []
        if test:
            data = self.test_data
        else:
            data = self.eval_data
        for i, episode in enumerate(data):
            if test:
                tnum = i
            else: 
                tnum = episode['number']
            num_timesteps = len(episode['timestep_list'])
            goal_dist = [episode['timestep_list'][j]['reward']['distance_to_goal'] for j in range(num_timesteps)]
            if min(goal_dist) <= 0.001:
                s_or_f.append(1)
                
                success_timesteps.append([tnum, num_timesteps])
                success_vel.append([tnum, np.linalg.norm(episode['timestep_list'][-1]['state']['obj_2']['velocity'])])
            else:
                s_or_f.append(0)
                fail_timesteps.append([i, num_timesteps])
                if num_timesteps < 400:
                    cut_short.append(num_timesteps)
                last_goal = episode['timestep_list'][-1]['reward']['goal_position']
                last_pos = episode['timestep_list'][-1]['state']['obj_2']['pose'][0]
                dxdy_final.append([tnum,abs(last_goal[0]-last_pos[0]),abs(last_goal[1]-last_pos[1])])
                closest = np.argmin(goal_dist)
                closest_goal = episode['timestep_list'][closest]['reward']['goal_position']
                closest_pos = episode['timestep_list'][closest]['state']['obj_2']['pose'][0]
                dxdy_closest.append([tnum,abs(closest_goal[0]-closest_pos[0]),abs(closest_goal[1]-closest_pos[1])])
        num_episodes = len(data)
        success_rate = moving_average(s_or_f, 10)
        plt.plot(range(len(success_rate)), success_rate)
        plt.title('Success Rate (10 ep moving average)')
        plt.xlabel('Episode')
        plt.ylabel('Success Rate')
        plt.show()
        plt.clf()
        success_timesteps = np.array(success_timesteps)
        success_vel = np.array(success_vel)
        fail_timesteps = np.array(fail_timesteps)
        dxdy_final = np.array(dxdy_final)
        dxdy_closest = np.array(dxdy_closest)
        if len(success_timesteps) > 0:
            plt.plot(success_timesteps[:,0], success_timesteps[:,1])
            plt.title('Num Timesteps in Successful Runs')
            plt.xlabel('Run number')
            plt.ylabel('Timesteps')
            plt.show()
            plt.clf()
            plt.plot(success_vel[:,0], success_vel[:,1])
            plt.title('Ending Speed in Successful Runs')
            plt.xlabel('Run Number')
            plt.ylabel('Speed')
            plt.show()
            plt.clf()
        if len(fail_timesteps) > 0: #HAHAHAHAHHAHHA
            plt.plot(fail_timesteps[:,0], fail_timesteps[:,1])
            plt.title('Num Timesteps in Failed Runs')
            plt.xlabel('Run number')
            plt.ylabel('Timesteps')
            plt.show()
            plt.clf()
            plt.plot(dxdy_final[:,0], dxdy_final[:,1])
            plt.plot(dxdy_final[:,0], dxdy_final[:,2])
            plt.title('Ending Distance from Goal in Failed Runs')
            plt.xlabel('Run Number')
            plt.ylabel('Distance From Goal')
            plt.legend(['dx','dy'])
            plt.show()
            plt.clf() 
            plt.plot(dxdy_closest[:,0], dxdy_closest[:,1])
            plt.plot(dxdy_closest[:,0], dxdy_closest[:,2])
            plt.title('Minimum Distance from Goal in Failed Runs')
            plt.xlabel('Run Number')
            plt.ylabel('Distance From Goal')
            plt.legend(['dx','dy'])
            plt.show()
            plt.clf() 
            print('when cut short, average length is ', np.average(cut_short))
            
    def plot_actions(self):
        actor_outputs = []
        actor_std = []
        for episode in self.test_data:
            num_timesteps = len(episode['timestep_list'])
            joint_angles = [episode['timestep_list'][i]['control']['actor_output'] for i in range(num_timesteps)]
            avg_angles = np.average(np.array(joint_angles), axis=0)
            std_angles = np.std(np.array(joint_angles), axis=0)
            actor_outputs.append(avg_angles)
            actor_std.append(std_angles)
        num_episodes = len(self.test_data)
        actor_outputs = np.array(actor_outputs)
        actor_std = np.array(actor_std)
        plt.errorbar(range(num_episodes), actor_outputs[:,0], actor_std[:,0])
        plt.errorbar(range(num_episodes), actor_outputs[:,1], actor_std[:,1])
        plt.errorbar(range(num_episodes), actor_outputs[:,2], actor_std[:,2])
        plt.errorbar(range(num_episodes), actor_outputs[:,3], actor_std[:,3])
        plt.xlabel('Episode')
        plt.ylabel('Actor Average Output')
        plt.title('Actor output (rad)')
        plt.legend(['Left Proximal', 'Left Distal', 'Right Proximal','Right Distal'])
        plt.show()        
        
stuff = plotMaker('/home/orochi/mojo/mojo-grasp/demos/rl_demo/data/vizualization/')
# stuff.plot_f1f2()
# stuff.plot_sf_stuff(False)
stuff.plot_actions()