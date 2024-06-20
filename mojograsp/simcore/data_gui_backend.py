#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 10:04:51 2023

@author: orochi
"""
import os
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import json
from scipy.stats import kde
import re
import sys
import time
from matplotlib.patches import Rectangle
from scipy.spatial.transform import Rotation as R
import mojograsp.simcore.reward_functions as rf
import multiprocessing
import pandas as pd

def getitem_for(d, key):
    for level in key:
        d = d[level]
    return d

def pool_process(episode_file):
    with open(episode_file, 'rb') as ef:
        tempdata = pkl.load(ef)
    data = tempdata['timestep_list']
    point_list = []
    try:
        point_list.extend([data[0]['state']['obj_2']['pose'][0][0],data[0]['state']['obj_2']['pose'][0][1]-0.1])
        point_list.extend([data[-1]['state']['obj_2']['pose'][0][0],data[-1]['state']['obj_2']['pose'][0][1]-0.1])
        point_list.extend(data[0]['state']['goal_pose']['goal_position'][0:2])
        point_list.append(data[0]['reward']['distance_to_goal'])
        point_list.append(data[-1]['reward']['distance_to_goal'])
        point_list.append(max([i['reward']['distance_to_goal'] for i in data]))
        point_list.append(data[-1]['reward']['object_orientation'][2]) # end orientation
        point_list.append(data[0]['state']['goal_pose']['goal_orientation']) # goal orientation
        point_list.append(episode_file)
    except KeyError:
        print('episode keys are wrong. check pool_process in data_gui_backend.py with your state and reward keys')

    return point_list

def pool_reward(episode_file,tholds, reward_func):
    with open(episode_file, 'rb') as ef:
        tempdata = pkl.load(ef)
    data = tempdata['timestep_list']
    return sum([reward_func(i['reward'],tholds)[0] for i in data])

def pool_key_list(episode_file, tsteps, key_tuples):
    '''
    This takes a given episode file, tuple of timestep numbers and tuple of key tuples and returns
    a list with each key at its desired timestep. Designed to be used with Pool.starmap on a folder 
    full of data
    '''
    # print(tsteps,key_tuples)
    with open(episode_file, 'rb') as ef:
        tempdata = pkl.load(ef)
    data = tempdata['timestep_list']
    return [getitem_for(data[tstep],key) for tstep, key in zip(tsteps, key_tuples)]

def pool_key_list_all(episode_file, key_tuples):
    with open(episode_file, 'rb') as ef:
        tempdata = pkl.load(ef)
    data = tempdata['timestep_list']
    desired = []
    for tup in key_tuples:
        desired.append([getitem_for(d,tup) for d in data])
    return desired

def goal_dist_process(episode_file, tstep):
    # print(tstep)
    with open(episode_file, 'rb') as ef:
        tempdata = pkl.load(ef)
    data = tempdata['timestep_list']
    t = [data[i]['reward']['distance_to_goal'] for i in range(tstep)]
    return min(t)

def moving_average(a, n) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

class PlotBackend():
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.clear_plots = True
        self.aspect_ratio = 3/4
        # self.fig.set_size_inches(12,9)4:3
        self.curr_graph = None
        self.moving_avg = 1 
        self.colorbar = None
        self.reduced_format = False
        self.config = {}
        self.legend = []
        self.point_dictionary = None
        self.tholds = []
        # self.load_config(config_folder)
        self.click_spell = None
        
    def load_config(self, config_folder):
        with open(config_folder+'/experiment_config.json') as file:
            self.config = json.load(file)

    def set_fig_size(self,width):
        self.fig.set_size_inches(width,width*self.aspect_ratio)

    def reset(self):
        self.click_spell = None
        self.legend = []
        self.point_dictionary = None
        self.tholds = []
        print('we just reset')

    def set_reward_func(self,key):
        # print(key)
        if key == 'Sparse':
            self.build_reward = rf.sparse
        elif key == 'Distance':
            self.build_reward = rf.distance
        elif key == 'Distance + Finger':
            self.build_reward = rf.distance_finger
        elif key == 'Hinge Distance + Finger':
            self.build_reward = rf.hinge_distance
        elif key == 'Slope':
            self.build_reward = rf.slope
        elif key == 'Slope + Finger':
            self.build_reward = rf.slope_finger
        elif key == 'SmartDistance + Finger':
            self.build_reward = rf.smart
        elif key == 'ScaledDistance + Finger':
            self.build_reward = rf.scaled
        elif key == "Rotation":
            self.build_reward = rf.rotation
        elif key == "solo_rotation":
            self.build_reward = rf.solo_rotation
        elif key == 'slide_and_rotate':
            self.build_reward = rf.slide_and_rotate
        elif key == 'rotation_with_finger':
            self.build_reward = rf.rotation_with_finger
        elif key == 'single_scaled':
            self.build_reward = rf.double_scaled
        elif key == 'SFS':
            self.build_reward = rf.sfs
        elif key == 'DFS':
            self.build_reward = rf.dfs
        elif key == 'TripleScaled':
            self.build_reward = rf.triple_scaled_slide
        elif key == 'SmartDistance + SmartFinger':
            self.build_reward = rf.double_smart
        elif key == 'multi_scaled':
            self.build_reward = rf.multi_scaled
        elif key =='contact point':
            self.build_reward = rf.contact_point
        elif key =='Rotation+Finger':
            self.build_reward = rf.rotation_with_finger
        else:
            raise Exception('reward type does not match list of known reward types')
    
    def set_tholds(self, tholds):
        '''expected thold format
        {'SUCCESS_THRESHOLD':float,
                       'DISTANCE_SCALING':float,
                       'CONTACT_SCALING':float,
                       'SUCCESS_REWARD':float}
        '''
        self.tholds = tholds


    def draw_path(self,data_dict):
        data = data_dict['timestep_list']
        episode_number=data_dict['number']
        trajectory_points = [f['state']['obj_2']['pose'][0] for f in data]
        print('goal position in state', data[0]['state']['goal_pose'])
        try:
            goal_poses = np.array([i['state']['goal_pose']['goal_pose'] for i in data])
        except:
            goal_poses = np.array([i['state']['goal_pose']['goal_position'] for i in data])
        # print(trajectory_points)
        trajectory_points = np.array(trajectory_points)
        ideal = np.zeros([len(goal_poses)+1,2])
        ideal[0,:] = trajectory_points[0,0:2]
        ideal[1:,:] = goal_poses + np.array([0,0.1])
        if self.clear_plots | (self.curr_graph != 'path'):
            self.clear_axes()
        self.ax.plot(trajectory_points[:,0], trajectory_points[:,1])
        self.ax.plot(ideal[:,0],ideal[:,1])
        self.ax.set_xlim([-0.08,0.08])
        self.ax.set_ylim([0.02,0.18])
        self.ax.set_xlabel('X pos (m)')
        self.ax.set_ylabel('Y pos (m)')                                                                                                                                                                                                                                   
        self.legend.extend(['RL Object Trajectory - episode '+str(episode_number), 'Ideal Path to Goal - episode '+str(episode_number)])
        self.ax.legend(self.legend)
        self.ax.set_title('Object Path')
        self.ax.set_aspect('equal',adjustable='box')
        self.curr_graph = 'path'
        # print(data[0]['state']['direction'])

    def draw_asterisk(self, folder_or_data_dict):
        
        # get list of pkl files in folder
        if type(folder_or_data_dict) is str:
            # print('need to load in episode all first')
            print('processing data from files in folder, this will be time consuming')
            episode_files = [os.path.join(folder_or_data_dict, f) for f in os.listdir(folder_or_data_dict) if f.lower().endswith('.pkl')]
            filenames_only = [f for f in os.listdir(folder_or_data_dict) if f.lower().endswith('.pkl')]
            
            filenums = [re.findall('\d+',f) for f in filenames_only]
            final_filenums = []
            for i in filenums:
                if len(i) > 0 :
                    final_filenums.append(int(i[0]))
            
            sorted_inds = np.argsort(final_filenums)
            final_filenums = np.array(final_filenums)
            episode_files = np.array(episode_files)
            filenames_only = np.array(filenames_only)
    
            episode_files = episode_files[sorted_inds].tolist()
            # filenames_only = filenames_only[sorted_inds].tolist()
    
            goals = []
            trajectories = []
            for episode_file in episode_files[-8:]:
                with open(episode_file, 'rb') as ef:
                    tempdata = pkl.load(ef)
                print(episode_file)
                data = tempdata['timestep_list']
                trajectory_points = [f['state']['obj_2']['pose'][0] for f in data]
                goal = data[1]['reward']['goal_position']
                goals.append(str(np.round(goal[0:2],2)))
                trajectories.append(np.array(trajectory_points))
        elif type(folder_or_data_dict) is dict:
            goals = []
            trajectories = []
            for episode in folder_or_data_dict['episode_list'][-8:]:
                data = episode['timestep_list']
                trajectory_points = [f['state']['obj_2']['pose'][0] for f in data]
                goal = data[1]['reward']['goal_position']
                goals.append(str(np.round(goal[0:2],2)))
                trajectories.append(np.array(trajectory_points))
        else:
            raise TypeError('First argument must be either folder containing episode data or dictionary containing all episode data')
        if self.clear_plots | (self.curr_graph != 'path'):
            self.clear_axes()
             
        
        self.ax.plot(trajectories[0][:,0], trajectories[0][:,1])
        self.ax.plot(trajectories[1][:,0], trajectories[1][:,1])
        self.ax.plot(trajectories[2][:,0], trajectories[2][:,1])
        self.ax.plot(trajectories[3][:,0], trajectories[3][:,1])
        self.ax.plot(trajectories[4][:,0], trajectories[4][:,1])
        self.ax.plot(trajectories[5][:,0], trajectories[5][:,1])
        self.ax.plot(trajectories[6][:,0], trajectories[6][:,1])
        self.ax.plot(trajectories[7][:,0], trajectories[7][:,1])
        self.ax.set_xlim([-0.07,0.07])
        self.ax.set_ylim([0.1,0.22])
        self.ax.set_xlabel('X pos (m)')
        self.ax.set_ylabel('Y pos (m)')                                                                                                                                                                                                                                   
        self.legend = goals
        self.ax.legend(self.legend)
        self.ax.set_title('Object Paths')
        self.curr_graph = 'path'
        self.ax.set_aspect('equal',adjustable='box')

    def draw_error(self,data_dict):
        
        data = data_dict['timestep_list']
        actor_output = np.array([f['action']['actor_output'] for f in data])
        f1_positions = np.array([f['state']['f1_pos'] for f in data])
        f2_positions = np.array([f['state']['f2_pos'] for f in data])
        
        actual_pos = [np.array([f1_positions[i+1][0],f1_positions[i+1][1],f2_positions[i+1][0],f2_positions[i+1][1]]) for i in range(len(f1_positions)-1)]
        
        ap = actor_output * 0.01       
        
        desired_pos = np.zeros(np.shape(actor_output))
        desired_pos[:,0:2] = f1_positions[:,0:2] + ap[:,0:2]
        desired_pos[:,2:4] = f2_positions[:,0:2] + ap[:,0:2]
        
        # actual_pos = np.array(actual_pos)
        errors = np.array([0,0,0,0])
        for i,poses in enumerate(actual_pos):
            errors += poses - desired_pos[i]
        print(errors)    
        
    def draw_angles(self, data_dict):
        data = data_dict['timestep_list']
        episode_number = data_dict['number']
        current_angle_dict = [f['state']['two_finger_gripper']['joint_angles'] for f in data]
        current_angle_list = []
        for angle in current_angle_dict:
            temp = [angs for angs in angle.values()]
            current_angle_list.append(temp)        
        # current_action_list= [f['action']['target_joint_angles'] for f in data]
        
        current_angle_list=np.array(current_angle_list)
        # current_action_list=np.array(current_action_list)
        angle_tweaks = current_angle_list
        if self.clear_plots | (self.curr_graph != 'angles'):
            self.clear_axes()
            
        self.ax.plot(range(len(angle_tweaks)),angle_tweaks[:,0])
        self.ax.plot(range(len(angle_tweaks)),angle_tweaks[:,1])
        self.ax.plot(range(len(angle_tweaks)),angle_tweaks[:,2])
        self.ax.plot(range(len(angle_tweaks)),angle_tweaks[:,3])
        self.legend.extend(['Right Proximal - episode '+str(episode_number), 
                            'Right Distal - episode '+str(episode_number), 
                            'Left Proximal - episode '+str(episode_number), 
                            'Left Distal - episode '+str(episode_number)])
        self.ax.legend(self.legend)
        self.ax.grid(True)
        self.ax.set_ylabel('Angle (radians)')
        self.ax.set_xlabel('Timestep (1/30 s)')
        self.ax.set_title('Joint Angles')
        self.curr_graph = 'angles'
        self.ax.set_aspect('auto',adjustable='box')
        
    def draw_actor_output(self, data_dict,action_type='FTP'):
        data = data_dict['timestep_list']
        episode_number = data_dict['number']
        actor_list = [f['action']['actor_output'] for f in data]
        actor_list = np.array(actor_list)
        if self.clear_plots | (self.curr_graph != 'angles'):
            self.clear_axes()
             
        self.ax.plot(range(len(actor_list)),actor_list[:,0])
        self.ax.plot(range(len(actor_list)),actor_list[:,1])
        self.ax.plot(range(len(actor_list)),actor_list[:,2])
        self.ax.plot(range(len(actor_list)),actor_list[:,3])
        if action_type == 'FTP':
            self.legend.extend(['Right X - episode ' + str( episode_number), 
                                'Right Y - episode ' + str( episode_number), 
                                'Left X - episode ' + str( episode_number), 
                                'Left Y - episode ' + str( episode_number)])
        elif action_type =='JA':
            self.legend.extend(['Right Proximal - episode '+str( episode_number), 
                                'Right Distal - episode '+str( episode_number), 
                                'Left Proximal - episode '+str( episode_number), 
                                'Left Distal - episode '+str( episode_number)])
        self.ax.legend(self.legend)
        self.ax.grid(True)
        self.ax.set_ylabel('Actor Output')
        self.ax.set_xlabel('Timestep (1/30 s)')
        self.ax.set_title('Actor Output')
        self.curr_graph = 'angles'
        self.ax.set_aspect('auto',adjustable='box')

    def draw_critic_output(self, data_dict):
        episode_number = data_dict['number']
        data = data_dict['timestep_list']
        try:
            critic_list = [f['control']['critic_output'] for f in data]
        except TypeError:
            print('pkl file does not contain critic data')
            return
        print(critic_list[0])
        if self.clear_plots | (self.curr_graph != 'rewards'):
            self.clear_axes()
             
        self.ax.plot(range(len(critic_list)),critic_list)
        self.legend.extend(['Critic Output - episode ' + str( episode_number)])
        self.ax.legend(self.legend)
        self.ax.grid(True)
        self.ax.set_ylabel('Action Value')
        self.ax.set_xlabel('Timestep (1/30 s)')
        self.ax.set_title('Critic Output')
        self.curr_graph = 'rewards'
        self.ax.set_aspect('auto',adjustable='box')

    def draw_object_distance(self, data_dict):
        episode_number = data_dict['number']

        data = data_dict['timestep_list']
        current_reward_dict = [f['reward']['distance_to_goal']*100 for f in data]

        if self.clear_plots | (self.curr_graph != 'rewards'):
            self.clear_axes()
             
        self.ax.plot(range(len(current_reward_dict)),current_reward_dict)
        self.legend.extend(['Distance to Goal - episode ' + str( episode_number)])
        self.ax.legend(self.legend)
        self.ax.grid(True)
        self.ax.set_ylabel('Distance (cm)')
        self.ax.set_xlabel('Timestep (1/30 s)')
        self.ax.set_title('Reward Plot')
         
        self.curr_graph = 'rewards'
        self.ax.set_aspect('auto',adjustable='box')
    
    def draw_contact_distance(self, data_dict):
        episode_number = data_dict['number']

        data = data_dict['timestep_list']
        current_reward_dict1 = [f['reward']['f1_dist']*100 for f in data]
        current_reward_dict2 = [f['reward']['f2_dist']*100 for f in data]
        if self.clear_plots | (self.curr_graph != 'rewards'):
            self.clear_axes()
             
        self.ax.plot(range(len(current_reward_dict1)),current_reward_dict1)
        self.ax.plot(range(len(current_reward_dict2)),current_reward_dict2)
        self.legend.extend(['Right Finger Contact Distance - episode ' + str( episode_number),'Left Finger Contact Distance - episode ' + str( episode_number)])
        self.ax.legend(self.legend)
        self.ax.grid(True)
        self.ax.set_ylabel('Distance (cm)')
        self.ax.set_xlabel('Timestep (1/30 s)')
        self.ax.set_title('Contact Reward Plot')
        self.ax.set_aspect('auto',adjustable='box')
        self.curr_graph = 'rewards'
        
    def draw_combined_rewards(self, data_dict):
        episode_number = data_dict['number']

        data = data_dict['timestep_list']
        full_reward = []
        general_reward = [f['reward'] for f in data]

        for reward_container in general_reward:
            temp = self.build_reward(reward_container, self.tholds)
            full_reward.append(temp[0])
        net_reward = sum(full_reward)
            
        if self.clear_plots | (self.curr_graph != 'rewards'):
            self.clear_axes()
        
        title = 'Net Reward: ' + str(net_reward)
        self.ax.plot(range(len(full_reward)),full_reward)
        self.legend.extend(['Reward - episode ' + str( episode_number)])
        self.ax.legend(self.legend)
        self.ax.set_ylabel('Reward')
        self.ax.set_xlabel('Timestep (1/30 s)')
        self.ax.set_title(title)
        self.ax.grid(True)
        self.ax.set_aspect('auto',adjustable='box')
        self.curr_graph = 'rewards'
         
    def draw_explored_region(self, folder):
        episode_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith('.pkl')]
        filenames_only = [f for f in os.listdir(folder) if f.lower().endswith('.pkl')]
        
        filenums = [re.findall('\d+',f) for f in filenames_only]
        final_filenums = []
        for i in filenums:
            if len(i) > 0 :
                final_filenums.append(int(i[0]))

        sorted_inds = np.argsort(final_filenums)
        final_filenums = np.array(final_filenums)
        episode_files = np.array(episode_files)
        filenames_only = np.array(filenames_only)
        episode_files = episode_files[sorted_inds].tolist()

        pool = multiprocessing.Pool()
        keys = (('state','obj_2','pose'),)
        thing = [[ef, keys] for ef in episode_files]
        print('applying async')
        data_list = pool.starmap(pool_key_list_all,thing)
        pool.close()
        pool.join()
        datapoints = []
        for i in data_list:
            # print(i)
            datapoints.extend([j[0][0:2] for j in i[0]])
        
        datapoints = np.array(datapoints)
        print('num poses', np.shape(datapoints))
        nbins=100
        x = datapoints[:,0]
        y = datapoints[:,1]
        print('about to do the gaussian')
        k = kde.gaussian_kde([x,y])
        print('did the gaussian')
        xlim = [np.min(datapoints[:,0]), np.max(datapoints[:,0])]
        ylim = [np.min(datapoints[:,1]), np.max(datapoints[:,1])]
        # xlim = [min(xlim[0],-0.07), max(xlim[1],0.07)]
        # ylim = [min(ylim[0],0.1), max(ylim[1],0.22)]
        xlim = [-0.1, 0.1]
        ylim = [0, 0.2]
        xi, yi = np.mgrid[xlim[0]:xlim[1]:nbins*1j, ylim[0]:ylim[1]:nbins*1j]
        print('did the mgrid')
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        term = -np.max(zi)/5
        for i in range(len(zi)):
            if zi[i] <1:
                zi[i] = term
        self.clear_axes()

        print('about to do the colormesh')
        c = self.ax.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto')

        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_xlabel('X pos (m)')
        self.ax.set_ylabel('Y pos (m)')
        self.ax.set_title("Explored Object Poses")
        self.ax.set_aspect('equal',adjustable='box')
        self.colorbar = self.fig.colorbar(c, ax=self.ax)
         
        self.curr_graph = 'explored'

    def draw_end_region(self, folder):
        episode_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith('.pkl')]
        filenames_only = [f for f in os.listdir(folder) if f.lower().endswith('.pkl')]
        
        filenums = [re.findall('\d+',f) for f in filenames_only]
        final_filenums = []
        for i in filenums:
            if len(i) > 0 :
                final_filenums.append(int(i[0]))

        sorted_inds = np.argsort(final_filenums)
        final_filenums = np.array(final_filenums)
        episode_files = np.array(episode_files)
        filenames_only = np.array(filenames_only)
        episode_files = episode_files[sorted_inds].tolist()

        '''
        datapoints = []
        for episode_file in episode_files:
            with open(episode_file, 'rb') as ef:
                tempdata = pkl.load(ef)

            data = tempdata['timestep_list']
            datapoints.append(data[-1]['state']['obj_2']['pose'][0][0:2])
        '''
        datapoints = [] 
        pool = multiprocessing.Pool()
        tst = (-1,)
        keys = (('state','obj_2','pose'),)
        thing = [[ef, tst, keys] for ef in episode_files]
        print('applying async')
        data_list = pool.starmap(pool_key_list,thing)
        pool.close()
        pool.join()
        for i in data_list:
            # print(i)
            datapoints.append(i[0][0][0:2])
        # print(datapoints)
        datapoints = np.array(datapoints)
        print('num poses', np.shape(datapoints))
        nbins=100
        x = datapoints[:,0]
        y = datapoints[:,1]
        print('about to do the gaussian')
        k = kde.gaussian_kde([x,y])
        print('did the gaussian')
        xlim = [np.min(datapoints[:,0]), np.max(datapoints[:,0])]
        ylim = [np.min(datapoints[:,1]), np.max(datapoints[:,1])]
        xlim = [-0.1, 0.1]
        ylim = [0.0, 0.2]
        xi, yi = np.mgrid[xlim[0]:xlim[1]:nbins*1j, ylim[0]:ylim[1]:nbins*1j]
        print('did the mgrid')
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        term = -np.max(zi)/5
        for i in range(len(zi)):
            if zi[i] <1:
                zi[i] = term
        self.clear_axes()
        self.ax.set_aspect('equal',adjustable='box')
        print('about to do the colormesh')
        c = self.ax.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto')

        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_xlabel('X pos (m)')
        self.ax.set_ylabel('Y pos (m)')
        self.ax.set_title("Sampled Object Poses")
        self.colorbar = self.fig.colorbar(c, ax=self.ax)
         
        self.curr_graph = 'ending_explored'
        
    def draw_sampled_region(self, filepath):
        with open(filepath+'/sampled_positions.pkl', 'rb') as pkl_file:
            datapoints = pkl.load(pkl_file)
        datapoints = np.array(datapoints)
        
        zi = datapoints
        start = time.time()
        nbins=100
        xlim = [-0.1, 0.1]
        ylim = [0.06, 0.26]
        xi, yi = np.mgrid[xlim[0]:xlim[1]:nbins*1j, ylim[0]:ylim[1]:nbins*1j]
        print('did the mgrid')
        term = -np.max(zi)/5
        print('about to do the thresholding')
        for i in range(len(zi)):
            for j in range(100):
                if zi[i,j] <1:
                    zi[i,j] = term
        self.clear_axes()
         
        print('about to do the colormesh')
        c = self.ax.pcolormesh(xi, yi, zi, shading='auto')
        end = time.time()
        print('total time:', end-start)
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_xlabel('X pos (m)')
        self.ax.set_ylabel('Y pos (m)')
        self.ax.set_title("explored object poses")
        self.ax.set_aspect('equal',adjustable='box')
        self.colorbar = self.fig.colorbar(c, ax=self.ax)
         
        self.curr_graph = 'explored'
    
    def draw_net_reward(self, folder_or_data_dict,plot_args=None):
        if type(folder_or_data_dict) is str:
            print('this will be slow, and we both know it')
            
            # get list of pkl files in folder
            episode_files = [os.path.join(folder_or_data_dict, f) for f in os.listdir(folder_or_data_dict) if f.lower().endswith('.pkl')]
            filenames_only = [f for f in os.listdir(folder_or_data_dict) if f.lower().endswith('.pkl')]
            
            filenums = [re.findall('\d+',f) for f in filenames_only]
            final_filenums = []
            for i in filenums:
                if len(i) > 0 :
                    final_filenums.append(int(i[0]))
            
            sorted_inds = np.argsort(final_filenums)
            final_filenums = np.array(final_filenums)
            episode_files = np.array(episode_files)
            filenames_only = np.array(filenames_only)

            episode_files = episode_files[sorted_inds].tolist()
            rewards = []
            count = 0
            for episode_file in episode_files:
                with open(episode_file, 'rb') as ef:
                    tempdata = pkl.load(ef)
                data = tempdata['timestep_list']
                individual_rewards = []
                for tstep in data:
                    individual_rewards.append(self.build_reward(tstep['reward'], self.tholds)[0])
                rewards.append(sum(individual_rewards))
                if count% 100 ==0:
                    print('count = ', count)
                count +=1
        elif type(folder_or_data_dict) is dict: 
            try:
                rewards = [-i['sum_dist']-i['sum_finger'] for i in folder_or_data_dict['episode_list']]
            except:
                rewards = []
                for episode in folder_or_data_dict['episode_list']:
                    data = episode['timestep_list']
                    individual_rewards = []
                    for timestep in data:
                        individual_rewards.append(self.build_reward(timestep['reward'], self.tholds)[0])
                    rewards.append(sum(individual_rewards))
        elif type(folder_or_data_dict) is list:
            rewards = folder_or_data_dict
        else:
            raise TypeError('argument should be string pointing to folder containing episode pickles, dictionary containing all episode data, or list of rewards')

        return_rewards = rewards.copy()
        if self.moving_avg != 1:
            rewards = moving_average(rewards,self.moving_avg)
        if self.clear_plots | (self.curr_graph !='Group_Reward'):
            self.clear_axes()
             
        self.legend.append('Average Agent Reward')
        if plot_args is None:
            self.ax.plot(range(len(rewards)), rewards)
        else:
            self.ax.plot(range(len(rewards)), rewards, color=plot_args[0], linestyle=plot_args[1])
        self.ax.set_xlabel('Episode Number')
        self.ax.set_ylabel('Average Total Reward Over the Entire Episode')
        self.ax.set_title("Agent Reward over Episode")
        # self.ax.set_ylim([-12, 0])
        self.ax.legend(self.legend)
        self.ax.grid(True)
        self.ax.set_aspect('auto',adjustable='box')
        self.curr_graph = 'Group_Reward'
        return return_rewards
        
    def draw_net_distance_reward(self, folder_or_data_dict):
        if type(folder_or_data_dict) is str:
            print('this will be slow, and we both know it')
            
            # get list of pkl files in folder
            episode_files = [os.path.join(folder_or_data_dict, f) for f in os.listdir(folder_or_data_dict) if f.lower().endswith('.pkl')]
            filenames_only = [f for f in os.listdir(folder_or_data_dict) if f.lower().endswith('.pkl')]
            
            filenums = [re.findall('\d+',f) for f in filenames_only]
            final_filenums = []
            for i in filenums:
                if len(i) > 0 :
                    final_filenums.append(int(i[0]))
            
            
            sorted_inds = np.argsort(final_filenums)
            final_filenums = np.array(final_filenums)
            temp = final_filenums[sorted_inds]
            episode_files = np.array(episode_files)
            filenames_only = np.array(filenames_only)

            episode_files = episode_files[sorted_inds].tolist()
            # filenames_only = filenames_only[sorted_inds].tolist()
            rewards = []
            temp = 0
            count = 0
            for episode_file in episode_files:
                with open(episode_file, 'rb') as ef:
                    tempdata = pkl.load(ef)
                data = tempdata['timestep_list']
                for timestep in data:
                    temp += - timestep['reward']['distance_to_goal']
                rewards.append(temp)
                temp = 0
                if count% 100 ==0:
                    print('count = ', count)
                count +=1
        elif type(folder_or_data_dict) is dict: 
            try:
                rewards = [-i['sum_dist'] for i in folder_or_data_dict['episode_list']]
            except:
                rewards = []
                temp = 0
                for episode in folder_or_data_dict['episode_list']:
                    data = episode['timestep_list']
                    for timestep in data:
                        temp += - timestep['reward']['distance_to_goal']
                    rewards.append(temp)
                    temp = 0
        elif type(folder_or_data_dict) is list:
            rewards = folder_or_data_dict
        else:
            raise TypeError('argument should be string pointing to folder containing episode pickles, dictionary containing all episode data, or list of rewards')
        return_rewards = rewards.copy()
        if self.moving_avg != 1:
            rewards = moving_average(rewards,self.moving_avg)
        if self.clear_plots | (self.curr_graph !='Group_Reward'):
            self.clear_axes()
             
        self.legend.append('Average Distance Reward')
        self.ax.plot(range(len(rewards)), rewards)
        self.ax.set_xlabel('Episode Number')
        self.ax.set_ylabel('Total Reward Over the Entire Episode')
        self.ax.set_title("Agent Reward over Episode")
        # self.ax.set_ylim([-12, 0])
        self.ax.legend(self.legend)
        self.ax.grid(True)
        self.ax.set_aspect('auto',adjustable='box')
        self.curr_graph = 'Group_Reward'
        return return_rewards

    def draw_finger_obj_dist_avg(self, folder_or_data_dict):

        if type(folder_or_data_dict) is str:
            print('this will be slow, and we both know it')
            
            # get list of pkl files in folder
            episode_files = [os.path.join(folder_or_data_dict, f) for f in os.listdir(folder_or_data_dict) if f.lower().endswith('.pkl')]
            filenames_only = [f for f in os.listdir(folder_or_data_dict) if f.lower().endswith('.pkl')]
            
            filenums = [re.findall('\d+',f) for f in filenames_only]
            final_filenums = []
            for i in filenums:
                if len(i) > 0 :
                    final_filenums.append(int(i[0]))
            
            sorted_inds = np.argsort(final_filenums)
            final_filenums = np.array(final_filenums)
            temp = final_filenums[sorted_inds]
            episode_files = np.array(episode_files)
            filenames_only = np.array(filenames_only)

            episode_files = episode_files[sorted_inds].tolist()

            pool = multiprocessing.Pool()
            keys = (('reward','f1_dist'),('reward','f2_dist'))
            thing = [[ef, keys] for ef in episode_files]
            print('applying async')
            data_list = pool.starmap(pool_key_list_all,thing)
            pool.close()
            pool.join()
            # filenames_only = filenames_only[sorted_inds].tolist()
            finger_obj_avgs= []
            for i in data_list:
                finger_obj_avgs.append([np.average(i[0]),np.average(i[1])])
            # print(finger_obj_avgs)
            finger_obj_avgs = np.array(finger_obj_avgs)
        
        if self.moving_avg != 1:
            t1 = moving_average(finger_obj_avgs[:,0],self.moving_avg)
            t2 = moving_average(finger_obj_avgs[:,1],self.moving_avg)
            finger_obj_avgs = np.transpose(np.array([t1,t2]))
        
        if self.clear_plots | (self.curr_graph != 'fing_obj_dist'):
            self.clear_axes()
                
        self.ax.plot(range(len(finger_obj_avgs)),finger_obj_avgs[:,0])
        self.ax.plot(range(len(finger_obj_avgs)),finger_obj_avgs[:,1])
        self.legend.extend(['Average Finger 1 Object Distance', 'Average Finger 2 Object Distance'])
        self.ax.legend(self.legend)
        self.ax.set_ylabel('Finger Object Distance')
        self.ax.set_xlabel('Episode')
        self.ax.grid(True)
        self.ax.set_title('Finger Object Distance Per Episode')
        self.ax.set_aspect('auto',adjustable='box')
        self.curr_graph = 'fing_obj_dist'

    def draw_finger_obj_dist_max(self, folder_or_data_dict):
        if type(folder_or_data_dict) is str:
            print('this will be slow, and we both know it')
            
            # get list of pkl files in folder
            episode_files = [os.path.join(folder_or_data_dict, f) for f in os.listdir(folder_or_data_dict) if f.lower().endswith('.pkl')]
            filenames_only = [f for f in os.listdir(folder_or_data_dict) if f.lower().endswith('.pkl')]
            
            filenums = [re.findall('\d+',f) for f in filenames_only]
            final_filenums = []
            for i in filenums:
                if len(i) > 0 :
                    final_filenums.append(int(i[0]))
            
            sorted_inds = np.argsort(final_filenums)
            final_filenums = np.array(final_filenums)
            temp = final_filenums[sorted_inds]
            episode_files = np.array(episode_files)
            filenames_only = np.array(filenames_only)

            episode_files = episode_files[sorted_inds].tolist()

            pool = multiprocessing.Pool()
            keys = (('reward','f1_dist'),('reward','f2_dist'))
            thing = [[ef, keys] for ef in episode_files]
            print('applying async')
            data_list = pool.starmap(pool_key_list_all,thing)
            pool.close()
            pool.join()
            finger_obj_maxs= []
            for i in data_list:
                finger_obj_maxs.append([np.max(i[0]),np.max(i[1])])
            finger_obj_maxs = np.array(finger_obj_maxs)

        if self.moving_avg != 1:
            t1 = moving_average(finger_obj_maxs[:,0],self.moving_avg)
            t2 = moving_average(finger_obj_maxs[:,1],self.moving_avg)
            finger_obj_maxs = np.transpose(np.array([t1,t2]))
        if self.clear_plots | (self.curr_graph != 'fing_obj_dist'):
            self.clear_axes()
        self.ax.plot(range(len(finger_obj_maxs)),finger_obj_maxs[:,0])
        self.ax.plot(range(len(finger_obj_maxs)),finger_obj_maxs[:,1])
        self.legend.extend(['Maximum Finger 1 Object Distance', 'Maximum Finger 2 Object Distance'])
        self.ax.legend(self.legend)
        self.ax.grid(True)
        self.ax.set_ylabel('Finger Object Distance')
        self.ax.set_xlabel('Episode')
        self.ax.set_title('Finger Object Distance Per Episode')
        self.ax.set_aspect('auto',adjustable='box')
        self.curr_graph = 'fing_obj_dist'
        
    def draw_timestep_bar_plot(self,all_data_dict):
        success_timesteps = []
        fail_timesteps = []
        for i, episode in enumerate(all_data_dict['episode_list']):
            data = episode['timestep_list']
            num_tsteps = len(data)
            ending_dist = data[-1]['reward']['distance_to_goal']
            if ending_dist < 0.002:
                success_timesteps.append([i, num_tsteps])
            else:
                fail_timesteps.append([i, num_tsteps])
        
        success_timesteps = np.array(success_timesteps)
        fail_timesteps = np.array(fail_timesteps)

        self.clear_axes()
                 
        self.ax.bar(fail_timesteps[:,0],fail_timesteps[:,1])
        if len(success_timesteps) > 0:
            self.ax.bar(success_timesteps[:,0],success_timesteps[:,1])
        self.legend.extend(['Failed Runs', 'Successful Runs'])
        self.ax.legend(self.legend)
        self.ax.set_ylabel('Number of Timesteps')
        self.ax.set_xlabel('Episode')
        self.ax.set_title('Number of Timesteps per Episode')
        self.ax.set_aspect('auto',adjustable='box')
         
    def draw_avg_actor_output(self, folder_or_data_dict, action_type='FTP'):
        if type(folder_or_data_dict) is str:
            episode_files = [os.path.join(folder_or_data_dict, f) for f in os.listdir(folder_or_data_dict) if (f.lower().endswith('.pkl') & ('all' not in f))]
            filenames_only = [f for f in os.listdir(folder_or_data_dict) if (f.lower().endswith('.pkl') & ('all' not in f))]
            
            filenums = [re.findall('\d+',f) for f in filenames_only]
            final_filenums = []
            for i in filenums:
                if len(i) > 0 :
                    final_filenums.append(int(i[0]))
            
            
            sorted_inds = np.argsort(final_filenums)
            final_filenums = np.array(final_filenums)
            episode_files = np.array(episode_files)
            filenames_only = np.array(filenames_only)

            episode_files = episode_files[sorted_inds].tolist()

            avg_actor_output = np.zeros((len(episode_files),4))
            avg_actor_std = np.zeros((len(episode_files),4))

            for i, episode_file in enumerate(episode_files):
                with open(episode_file, 'rb') as ef:
                    tempdata = pkl.load(ef)
                data = tempdata['timestep_list']
                actor_list = [f['action']['actor_output'] for f in data]
                actor_list = np.array(actor_list)
                # print(actor_list)
                # print(np.average(actor_list, axis=1))
                avg_actor_output[i,:] = np.average(actor_list, axis = 0)
                avg_actor_std[i,:] = np.std(actor_list, axis = 0)
                if i % 100 ==0:
                    print('count = ',i)
        elif type(folder_or_data_dict) is dict:
            avg_actor_output = np.zeros((len(folder_or_data_dict['episode_list']),4))
            avg_actor_std = np.zeros((len(folder_or_data_dict['episode_list']),4))
            for i, episode in enumerate(folder_or_data_dict['episode_list']):
                data = episode['timestep_list']
                actor_list = [f['control']['actor_output'] for f in data]
                actor_list = np.array(actor_list)
                # print(actor_list)
                # print(np.average(actor_list, axis=1))
                avg_actor_output[i,:] = np.average(actor_list, axis = 0)
                avg_actor_std[i,:] = np.std(actor_list, axis = 0)
        else:
            raise TypeError('arguemnt should be string containing filepath with episode data or dictionary containing episode data')
        if self.moving_avg != 1:
            t1 = moving_average(avg_actor_output[:,0],self.moving_avg)
            t2 = moving_average(avg_actor_output[:,1],self.moving_avg)
            t3 = moving_average(avg_actor_output[:,2],self.moving_avg)
            t4 = moving_average(avg_actor_output[:,3],self.moving_avg)
            avg_actor_output = np.transpose(np.array([t1,t2,t3,t4]))
            t1 = moving_average(avg_actor_std[:,0],self.moving_avg)
            t2 = moving_average(avg_actor_std[:,1],self.moving_avg)
            t3 = moving_average(avg_actor_std[:,2],self.moving_avg)
            t4 = moving_average(avg_actor_std[:,3],self.moving_avg)
            avg_actor_std = np.transpose(np.array([t1,t2,t3,t4]))
                        
        
        if self.clear_plots | (self.curr_graph != 'angles_total'):
            self.clear_axes()
             
        self.ax.errorbar(range(len(avg_actor_output)),avg_actor_output[:,0], avg_actor_std[:,0])
        self.ax.errorbar(range(len(avg_actor_output)),avg_actor_output[:,1], avg_actor_std[:,1])
        self.ax.errorbar(range(len(avg_actor_output)),avg_actor_output[:,2], avg_actor_std[:,2])
        self.ax.errorbar(range(len(avg_actor_output)),avg_actor_output[:,3], avg_actor_std[:,3])
        if action_type == 'FTP':
            self.legend.extend(['Right X', 
                                'Right Y', 
                                'Left X', 
                                'Left Y'])
        elif action_type == 'JA':            
            self.legend.extend(['Right Proximal', 
                                'Right Distal', 
                                'Left Proximal', 
                                'Left Distal'])
        self.ax.legend(self.legend)
        self.ax.set_ylabel('Actor Output')
        self.ax.set_xlabel('Episode')
        self.ax.grid(True)
        self.ax.set_title('Actor Output')
        self.ax.set_aspect('auto',adjustable='box')
        self.curr_graph = 'angles_total'    

    def draw_shortest_goal_dist(self, folder_or_data_dict):
        if type(folder_or_data_dict) is str:
            # get list of pkl files in folder
            episode_files = [os.path.join(folder_or_data_dict, f) for f in os.listdir(folder_or_data_dict) if f.lower().endswith('.pkl')]
            filenames_only = [f for f in os.listdir(folder_or_data_dict) if f.lower().endswith('.pkl')]
            
            filenums = [re.findall('\d+',f) for f in filenames_only]
            final_filenums = []
            for i in filenums:
                if len(i) > 0 :
                    final_filenums.append(int(i[0]))
            
            sorted_inds = np.argsort(final_filenums)
            final_filenums = np.array(final_filenums)
            temp = final_filenums[sorted_inds]
            episode_files = np.array(episode_files)
            filenames_only = np.array(filenames_only)

            episode_files = episode_files[sorted_inds].tolist()
            
            min_dists = []
            short_names = []
            for i, episode_file in enumerate(episode_files):
                with open(episode_file, 'rb') as ef:
                    tempdata = pkl.load(ef)
                data = tempdata['timestep_list']
                goal_dists = [f['reward']['distance_to_goal'] for f in data]
                min_dists.append(min(goal_dists))
                if min(goal_dists) < 0.004:
                    short_names.append(episode_file)
                if i % 100 ==0:
                    print('count = ',i)
        elif type(folder_or_data_dict) is dict:
            try:
                min_dists = [i['min_dist'] for i in folder_or_data_dict['episode_list']]
            except:
                min_dists = np.zeros((len(folder_or_data_dict['episode_list']),1))
                for i, episode in enumerate(folder_or_data_dict['episode_list']):
                    data = episode['timestep_list']
                    goal_dist = np.zeros(len(data))
                    for j, timestep in enumerate(data):
                        goal_dist[j] = timestep['reward']['distance_to_goal']
                    min_dists[i] = np.min(goal_dist, axis=0)
        elif type(folder_or_data_dict) is list:
            min_dists = folder_or_data_dict
        else:
            raise TypeError('argument should be string pointing to folder containing episode pickles, dictionary containing all episode data, or list of min dists')

        return_mins = min_dists.copy()
        if self.moving_avg != 1:
            min_dists = moving_average(min_dists,self.moving_avg)
        if self.clear_plots | (self.curr_graph != 'goal_dist'):
            self.clear_axes()
             
        print(short_names)
        self.ax.plot(range(len(min_dists)),min_dists)
        self.legend.extend(['Min Goal Distance'])
        self.ax.legend(self.legend)
        self.ax.set_ylabel('Goal Distance')
        self.ax.set_xlabel('Episode')
        self.ax.grid(True)
        self.ax.set_title('Distance to Goal Per Episode')
        self.ax.set_aspect('auto',adjustable='box')
        self.curr_graph = 'goal_dist'
        return return_mins
            
    def draw_ending_velocity(self, all_data_dict):

        velocity = np.zeros((len(all_data_dict['episode_list']),2))
        for i, episode in enumerate(all_data_dict['episode_list']):
            data = episode['timestep_list']
            obj_poses = [f['state']['obj_2']['pose'][0] for f in data[-5:]]
            obj_poses = np.array(obj_poses)
            dx = obj_poses[-1,0] - obj_poses[0,0]
            dy = obj_poses[-1,1] - obj_poses[0,1]
            velocity[i,:] = [dx,dy]

        if self.clear_plots | (self.curr_graph != 's_f'):
            self.clear_axes()
             
        ending_vel = np.sqrt(velocity[:,0]**2 + velocity[:,1]**2)
        if self.moving_avg != 1:
            ending_vel = moving_average(ending_vel,self.moving_avg)
        self.clear_axes()
                 
        self.ax.plot(range(len(ending_vel)),ending_vel)
        self.ax.set_ylabel('ending velocity magnitude')
        self.ax.set_xlabel('episode')
        self.ax.grid(True)
        self.ax.set_title("Ending Velocity")
        self.ax.set_aspect('auto',adjustable='box')
        self.curr_graph = 'vel'

    def draw_success_rate(self, folder_or_data_dict, success_range, rot_success_range):
        if type(folder_or_data_dict) is str:
            episode_files = [os.path.join(folder_or_data_dict, f) for f in os.listdir(folder_or_data_dict) if f.lower().endswith('.pkl')]
            filenames_only = [f for f in os.listdir(folder_or_data_dict) if f.lower().endswith('.pkl')]
            
            filenums = [re.findall('\d+',f) for f in filenames_only]
            final_filenums = []
            for i in filenums:
                if len(i) > 0 :
                    final_filenums.append(int(i[0]))
            
            
            sorted_inds = np.argsort(final_filenums)
            final_filenums = np.array(final_filenums)
            episode_files = np.array(episode_files)
            filenames_only = np.array(filenames_only)
            count = 0
            episode_files = episode_files[sorted_inds].tolist()
            ending_dists = []
            ending_orientation_errors = []
            pool = multiprocessing.Pool()
            tst = (-1,-1,-1)
            keys = (('reward','distance_to_goal'),('state','obj_2','z_angle'),('state','goal_pose','goal_orientation'))
            thing = [[ef, tst, keys] for ef in episode_files]
            print('applying async')
            data_list = pool.starmap(pool_key_list,thing)
            pool.close()
            pool.join()
            for i in data_list:
                ending_dists.append(i[0])
                ending_orientation_errors.append(i[2]-i[1])
            ending_dists = np.array(ending_dists)
            ending_orientation_errors = np.array(ending_orientation_errors)
            ending_orientation_errors = np.abs(ending_orientation_errors) *180 /np.pi
        elif type(folder_or_data_dict) is dict:
            try:
                ending_dists = [i['ending_dist'] for i in folder_or_data_dict['episode_list']]
            except:
                ending_dists = np.zeros((len(folder_or_data_dict['episode_list']),1))
                for i, episode in enumerate(folder_or_data_dict['episode_list']):
                    data = episode['timestep_list']
                    ending_dists[i] = np.max(data[-1]['reward']['distance_to_goal'], axis=0)
        elif type(folder_or_data_dict) is list:
            ending_dists = folder_or_data_dict
        else:
            raise TypeError('argument should be string pointing to folder containing episode pickles, dictionary containing all episode data, or list of ending dists')
        s_f = []
        for dist, orr in zip(ending_dists,ending_orientation_errors):
            if (dist < success_range/1000) and (orr < rot_success_range):
                s_f.append(100)
            else:
                s_f.append(0)
        return_dists = ending_dists.copy()
        print('total success rate', np.average(s_f))
        if self.moving_avg != 1:
            s_f = moving_average(s_f,self.moving_avg)
        if self.clear_plots | (self.curr_graph != 's_f'):
            self.clear_axes()
             
        self.ax.plot(range(len(s_f)),s_f)
        self.legend.extend(['Success Rate (' + str(success_range) + ' mm tolerance)'])
        self.ax.legend(self.legend)
        self.ax.set_ylabel('Success Percentage')
        self.ax.set_xlabel('Episode')
        self.ax.set_ylim([-1,101])
        titlething = 'Percent of Trials over ' + str(self.moving_avg)+' window that are successful'
        self.ax.set_title(titlething)
        self.ax.grid(True)
        self.ax.set_aspect('auto',adjustable='box')
        self.curr_graph = 's_f'
        
        return np.average(s_f)

    def draw_ending_goal_dist(self, folder_or_data_dict):
        if type(folder_or_data_dict) is str:
            episode_files = [os.path.join(folder_or_data_dict, f) for f in os.listdir(folder_or_data_dict) if f.lower().endswith('.pkl')]
            filenames_only = [f for f in os.listdir(folder_or_data_dict) if f.lower().endswith('.pkl')]
            
            filenums = [re.findall('\d+',f) for f in filenames_only]
            final_filenums = []
            for i in filenums:
                if len(i) > 0 :
                    final_filenums.append(int(i[0]))
            
            
            sorted_inds = np.argsort(final_filenums)
            final_filenums = np.array(final_filenums)
            episode_files = np.array(episode_files)
            filenames_only = np.array(filenames_only)
            count = 0
            episode_files = episode_files[sorted_inds].tolist()
            ending_dists = []
            names=[]
            for episode_file in episode_files:
                with open(episode_file, 'rb') as ef:
                    tempdata = pkl.load(ef)

                data = tempdata['timestep_list']
                names.append(tempdata['hand name'])
                ending_dists.append(data[-1]['reward']['distance_to_goal'])
                if count% 100 ==0:
                    print('count = ', count)
                count +=1
            print('unique names', np.unique(names))
        elif type(folder_or_data_dict) is dict:
            try:
                ending_dists = [i['ending_dist'] for i in folder_or_data_dict['episode_list']]
            except:
                ending_dists = np.zeros((len(folder_or_data_dict['episode_list']),1))
                for i, episode in enumerate(folder_or_data_dict['episode_list']):
                    data = episode['timestep_list']
                    ending_dists[i] = np.max(data[-1]['reward']['distance_to_goal'], axis=0)
        elif type(folder_or_data_dict) is list:
            ending_dists = folder_or_data_dict
        else:
            raise TypeError('argument should be string pointing to folder containing episode pickles, dictionary containing all episode data, or list of ending dists')

        mean, std = np.average(ending_dists), np.std(ending_dists)


        if self.moving_avg != 1:
            ending_dists = moving_average(ending_dists,self.moving_avg)
            
        if self.clear_plots | (self.curr_graph != 'goal_dist'):
            self.clear_axes()
             
                
        self.ax.plot(range(len(ending_dists)),ending_dists)
        self.legend.extend(['Ending Goal Distance'])
        self.ax.legend(self.legend)
        self.ax.set_ylabel('Goal Distance')
        self.ax.set_xlabel('Episode')
        self.ax.set_title('Distance to Goal Per Episode')
        self.ax.grid(True)
        self.ax.set_aspect('auto',adjustable='box')
        self.curr_graph = 'goal_dist'
        return [mean, std]
        
    def draw_goal_rewards(self, all_data_dict): # Depreciated
        
        keylist = ['forward','backward', 'forwardleft', 'backwardleft','forwardright', 'backwardright', 'left', 'right']
        rewards = {'forward':[[0.0, 0.2],[]],'backward':[[0.0, 0.12],[]],'forwardleft':[[-0.03, 0.19],[]],'backwardleft':[[-0.03,0.13],[]],
                   'forwardright':[[0.03,0.19],[]],'backwardright':[[0.03, 0.13],[]],'left':[[-0.04, 0.16],[]],'right':[[0.04,0.16],[]]}
        # sucessful_dirs = []
        for i, episode in enumerate(all_data_dict['episode_list']):
            data = episode['timestep_list']
            goal_pose = data[1]['reward']['goal_position'][0:2]
            temp = 0
            for j, timestep in enumerate(data):
                temp += - timestep['reward']['distance_to_goal'] \
                        -max(timestep['reward']['f1_dist'],timestep['reward']['f2_dist'])/5
                                        #timestep['reward']['end_penalty'] \
            for i, v in rewards.items():
                if np.isclose(goal_pose, v[0]).all():
                    v[1].append(temp)
                    
        # s = np.unique(sucessful_dirs)
        # print('succesful directions', s)
        maxes = []
        for name, v in rewards.items():
            print(name, max(v[1]))
            maxes.append(max(v[1]))
        print('showing best and worse ones')
        best = np.argmax(maxes)
        worst = np.argmin(maxes)
        
        # if self.moving_avg != 1:
        #     closest_dists = moving_average(closest_dists,self.moving_avg)
        if self.clear_plots | (self.curr_graph != 'goal_dist'):
            self.clear_axes()
             
            
        self.ax.plot(range(len(rewards[keylist[best]][1])),rewards[keylist[best]][1])
        self.ax.plot(range(len(rewards[keylist[worst]][1])),rewards[keylist[worst]][1])
        self.legend.extend(['Best Direction: ' + keylist[best], 'Worst Direction: ' + keylist[worst]])
        self.ax.legend(self.legend)
        self.ax.set_ylabel('Net Reward')
        self.ax.set_xlabel('Episode')
        self.ax.set_title('Net Reward By Direction')
        self.ax.grid(True)
        self.ax.set_aspect('auto',adjustable='box')
        self.curr_graph = 'direction_success_thing' 

    def draw_actor_max_percent(self, folder_path, action_type='FTP'):
        # get list of pkl files in folder
        episode_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith('.pkl')]
        filenames_only = [f for f in os.listdir(folder_path) if f.lower().endswith('.pkl')]
        
        filenums = [re.findall('\d+',f) for f in filenames_only]
        final_filenums = []
        for i in filenums:
            if len(i) > 0 :
                final_filenums.append(int(i[0]))
        
        
        sorted_inds = np.argsort(final_filenums)
        final_filenums = np.array(final_filenums)
        episode_files = np.array(episode_files)
        filenames_only = np.array(filenames_only)

        episode_files = episode_files[sorted_inds].tolist()
        
        pool = multiprocessing.Pool()
        keys = (('action','actor_output'),)
        thing = [[ef, keys] for ef in episode_files]
        print('applying async')
        data_list = pool.starmap(pool_key_list_all,thing)
        pool.close()
        pool.join()
        actor_max = []
        for i in data_list:
            actor_list = np.abs(np.array(i))>0.99
            end_actor = np.sum(actor_list,axis=0)/len(actor_list)
            actor_max.append(end_actor)


        actor_max = np.array(actor_max)
        if self.clear_plots | (self.curr_graph != 'angles'):
            self.clear_axes()
             
        self.ax.plot(range(len(actor_max)),actor_max[:,0])
        self.ax.plot(range(len(actor_max)),actor_max[:,1])
        self.ax.plot(range(len(actor_max)),actor_max[:,2])
        self.ax.plot(range(len(actor_max)),actor_max[:,3])
        if action_type =='FTP':
            self.legend.extend(['Right X Percent at Max', 
                                'Right Y Percent at Max', 
                                'Left X Percent at Max', 
                                'Left Y Percent at Max'])
        elif action_type=='JA':            
            self.legend.extend(['Right Proximal Percent at Max', 
                                'Right Distal Percent at Max', 
                                'Left Proximal Percent at Max', 
                                'Left Distal Percent at Max'])
        self.ax.legend(self.legend)
        self.ax.grid(True)
        self.ax.set_ylabel('Actor Output')
        self.ax.set_xlabel('Timestep (1/30 s)')
        self.ax.set_title('Fraction of Episode that Action is Maxed')
        self.ax.set_aspect('auto',adjustable='box')
        self.curr_graph = 'angles'
    
    def draw_goal_s_f(self, all_data_dict, success_range): # Depreciated

        rewards = {'forward':[[0.0, 0.2],[]],'backward':[[0.0, 0.12],[]],'forwardleft':[[-0.03, 0.19],[]],'backwardleft':[[-0.03,0.13],[]],
                   'forwardright':[[0.03,0.19],[]],'backwardright':[[0.03, 0.13],[]],'left':[[-0.04, 0.16],[]],'right':[[0.04,0.16],[]]}
        # sucessful_dirs = []
        for i, episode in enumerate(all_data_dict['episode_list']):
            data = episode['timestep_list']
            goal_dists = [f['reward']['distance_to_goal'] for f in data]
            ending_dist = min(goal_dists)
            goal_pose = data[1]['reward']['goal_position'][0:2]
            if ending_dist < success_range:
                temp = 100
            else:
                temp = 0
            for i, v in rewards.items():
                if np.isclose(goal_pose, v[0]).all():
                    v[1].append(temp)

        sf = []
        reduced_key_list = ['forward','backward','left','right']
        if self.moving_avg != 1:
            for i in reduced_key_list:
                sf.append(moving_average(rewards[i][1],self.moving_avg))
        if self.clear_plots | (self.curr_graph != 's_f'):
            self.clear_axes()
             
        for i,yax in enumerate(sf):
            self.ax.plot(range(len(yax)),yax)
            self.legend.extend([reduced_key_list[i]])
        self.ax.legend(self.legend)
        self.ax.set_ylabel('Net Reward')
        self.ax.set_xlabel('Episode')
        self.ax.set_title('Net Reward By Direction')
        self.ax.grid(True)
        self.ax.set_aspect('auto',adjustable='box')
        self.curr_graph = 'direction_reward_thing'   
    
    def draw_fingertip_path(self, data_dict):
        data = data_dict['timestep_list']
        episode_number = data_dict['number']
        trajectory_points = [f['state']['obj_2']['pose'][0] for f in data]
        fingertip1_points = [f['state']['f1_pos'] for f in data]
        fingertip2_points = [f['state']['f2_pos'] for f in data]
        goal_pose = data[1]['reward']['goal_position']
        trajectory_points = np.array(trajectory_points)
        fingertip1_points = np.array(fingertip1_points)
        fingertip2_points = np.array(fingertip2_points)
        arrow_len = max(int(len(trajectory_points)/25),1)
        arrow_points = np.linspace(0,len(trajectory_points)-arrow_len-1,10,dtype=int)
        next_points = arrow_points + arrow_len

        self.clear_axes()
         
        self.ax.plot(trajectory_points[:,0], trajectory_points[:,1])
        self.ax.plot(fingertip1_points[:,0], fingertip1_points[:,1])
        self.ax.plot(fingertip2_points[:,0], fingertip2_points[:,1])
        self.ax.plot([trajectory_points[0,0], goal_pose[0]],[trajectory_points[0,1],goal_pose[1]],marker='o')
        self.ax.plot(fingertip1_points[0,0], fingertip1_points[0,1], marker='o',
                     markersize=5,markerfacecolor='orange',markeredgecolor='orange')
        self.ax.plot(fingertip2_points[0,0], fingertip2_points[0,1], marker='o',
                     markersize=5,markerfacecolor='green',markeredgecolor='green')
        # self.ax.plot(finger_contact1[0], finger_contact1[1], marker='s', markersize=5,
        #              markerfacecolor='darkorange',markeredgecolor='darkorange')
        # self.ax.plot(finger_contact2[0], finger_contact2[1], marker='s', markersize=5,
        #              markerfacecolor='darkgreen',markeredgecolor='darkgreen')
        self.ax.set_xlim([-0.08,0.08])
        self.ax.set_ylim([0.02,0.18])
        self.ax.set_xlabel('X pos (m)')
        self.ax.set_ylabel('Y pos (m)')
        for i,j in zip(arrow_points,next_points):
            self.ax.arrow(trajectory_points[i,0],trajectory_points[i,1], 
                          trajectory_points[j,0]-trajectory_points[i,0],
                          trajectory_points[j,1]-trajectory_points[i,1], 
                          color='blue', width=0.001, head_width = 0.002, length_includes_head=True)
            self.ax.arrow(fingertip1_points[i,0],fingertip1_points[i,1], 
                          fingertip1_points[j,0]-fingertip1_points[i,0],
                          fingertip1_points[j,1]-fingertip1_points[i,1],
                          color='orange', width=0.001, head_width = 0.002, length_includes_head=True)
            self.ax.arrow(fingertip2_points[i,0],fingertip2_points[i,1], 
                          fingertip2_points[j,0]-fingertip2_points[i,0],
                          fingertip2_points[j,1]-fingertip2_points[i,1],
                          color='green', width=0.001, head_width = 0.002, length_includes_head=True)
            
        # self.ax.add_patch(Rectangle((0-0.038/2, 0.1-0.038/2), 0.038, 0.038,edgecolor='black',facecolor='white',alpha=0.5))
        self.legend.extend(['Object Trajectory','Right Finger Trajectory',
                            'Left Finger Trajectory','Ideal Path to Goal'])
        self.ax.legend(self.legend)
        self.ax.set_title('Object and Finger Path - Episode: '+str(episode_number))
        self.ax.set_aspect('equal',adjustable='box')
        self.curr_graph = 'path'

    def draw_obj_contacts(self, data_dict):
        episode_number = data_dict['number']

        data = data_dict['timestep_list']
        trajectory_points = [f['state']['obj_2']['pose'][0] for f in data]
        fingertip1_points = [f['state']['f1_pos'] for f in data if f['number']%5==1]
        fingertip2_points = [f['state']['f2_pos'] for f in data if f['number']%5==1]
        finger_contact1 = [f['state']['f1_contact_pos'] for f in data if f['number']%5==1]
        finger_contact2 = [f['state']['f2_contact_pos'] for f in data if f['number']%5==1]
        finger_dist1 = [f['reward']['f1_dist'] for f in data if f['number']%5==1]
        finger_dist2 = [f['reward']['f2_dist'] for f in data if f['number']%5==1]

        finger_contact1 = np.array(finger_contact1)
        finger_contact2 = np.array(finger_contact2)

        finger_dist1 = np.array(finger_dist1)
        finger_dist2 = np.array(finger_dist2)
        goal_pose = data[1]['reward']['goal_position']
        trajectory_points = np.array(trajectory_points)
        fingertip1_points = np.array(fingertip1_points)
        fingertip2_points = np.array(fingertip2_points)

        end_rot = R.from_quat(data[-1]['state']['obj_2']['pose'][1])
        end_rot = end_rot.as_euler('xyz', degrees=True)
        
        print(end_rot,data[-1]['state']['obj_2']['pose'][1])
        
        obase = np.array([1.0, 0.6470588235294118, 0.0])
        rbase = np.array([1.0, 0.0, 0.0])
        mid = np.array([1.0, 0.32, 0.0])
        gbase =  np.array([0.0, 0.5019607843137255, 0.0])
        midb =  np.array([0.0, 0.25, 0.5])
        bbase = np.array([0,0,1])
        
        f1_color_mapping = []
        f2_color_mapping = []
        for f1,f2 in zip(finger_dist1, finger_dist2):
            if f1 < 0.001:
                f1_color_mapping.append(obase)
            elif f1 > 0.005:
                f1_color_mapping.append(rbase)
            else:
                f1_color_mapping.append(mid)
            if f2 < 0.001:
                f2_color_mapping.append(gbase)
            elif f2 > 0.005:
                f2_color_mapping.append(bbase)
            else:
                f2_color_mapping.append(midb)
        print(f1_color_mapping, f2_color_mapping)
        print(finger_dist1, finger_dist2)
        self.clear_axes()
         
        self.ax.plot(trajectory_points[:,0], trajectory_points[:,1])
        self.ax.plot([trajectory_points[0,0], goal_pose[0]],[trajectory_points[0,1],goal_pose[1]],
                     c='red')
        self.ax.scatter(fingertip1_points[:,0], fingertip1_points[:,1], marker='+',
                     s=5,c='orange')
        self.ax.scatter(fingertip2_points[:,0], fingertip2_points[:,1], marker='+',
                     s=5,c='green')
        self.ax.scatter(finger_contact1[:,0], finger_contact1[:,1], marker='s', s=5,
                     c=f1_color_mapping)
        self.ax.scatter(finger_contact2[:,0], finger_contact2[:,1], marker='s', s=5,
                     c=f2_color_mapping)
        self.ax.set_xlim([-0.07,0.07])
        self.ax.set_ylim([0.04,0.16])
        self.ax.set_xlabel('X pos (m)')
        self.ax.set_ylabel('Y pos (m)')

        self.ax.add_patch(Rectangle((0-0.038/2, 0.1-0.038/2), 0.038, 0.038,edgecolor='black',facecolor='white',alpha=0.5))
        self.ax.add_patch(Rectangle((trajectory_points[-1,0]-0.038/2, 
                                     trajectory_points[-1,1]-0.038/2), 0.038, 0.038,
                                    edgecolor='black',facecolor='white',alpha=0.5, 
                                    angle=end_rot[-1],rotation_point='center'))
        self.legend.extend(['Object Trajectory','Ideal Path to Goal',
                            'Right Fingertip','Left Fingertip','Right Contact','Left Contact'])
        self.ax.legend(self.legend)
        self.ax.set_title('Object and Finger Path - Episode: '+str( episode_number))
        self.ax.set_aspect('equal',adjustable='box')
        self.curr_graph = 'path'

    def draw_net_finger_reward(self, folder_or_data_dict):
        if type(folder_or_data_dict) is str:
            print('this will be slow, and we both know it')
            
            # get list of pkl files in folder
            episode_files = [os.path.join(folder_or_data_dict, f) for f in os.listdir(folder_or_data_dict) if f.lower().endswith('.pkl')]
            filenames_only = [f for f in os.listdir(folder_or_data_dict) if f.lower().endswith('.pkl')]
            
            filenums = [re.findall('\d+',f) for f in filenames_only]
            final_filenums = []
            for i in filenums:
                if len(i) > 0 :
                    final_filenums.append(int(i[0]))
            
            sorted_inds = np.argsort(final_filenums)
            final_filenums = np.array(final_filenums)
            temp = final_filenums[sorted_inds]
            episode_files = np.array(episode_files)
            filenames_only = np.array(filenames_only)

            episode_files = episode_files[sorted_inds].tolist()
            rewards = []
            temp = 0
            count = 0
            for episode_file in episode_files:
                with open(episode_file, 'rb') as ef:
                    tempdata = pkl.load(ef)
                data = tempdata['timestep_list']
                for timestep in data:
                    temp += max(timestep['reward']['f1_dist'],timestep['reward']['f2_dist'])
                rewards.append(temp)
                temp = 0
                if count% 100 ==0:
                    print('count = ', count)
                count +=1
        elif type(folder_or_data_dict) is dict: 
            try:
                rewards = [-i['sum_finger'] for i in folder_or_data_dict['episode_list']]
                print('new one')
            except:
                rewards = []
                temp = 0
                for episode in folder_or_data_dict['episode_list']:
                    data = episode['timestep_list']
                    for timestep in data:
                        temp += max(timestep['reward']['f1_dist'],timestep['reward']['f2_dist'])
                    rewards.append(temp)
                    temp = 0
        elif type(folder_or_data_dict) is list:
            rewards = folder_or_data_dict
        else:
            raise TypeError('argument should be a string containing a path to the episode files, a dictionary containing the relevant data or a list containing the finger rewards')
        return_finger_rewards = rewards.copy()
        if self.moving_avg != 1:
            rewards = moving_average(rewards,self.moving_avg)
        if self.clear_plots | (self.curr_graph !='Group_Reward'):
            self.clear_axes()
         
        self.ax.plot(range(len(rewards)), rewards)
        self.ax.set_xlabel('Episode Number')
        self.ax.set_ylabel('Total Reward Over the Entire Episode')
        self.ax.set_title("Agent Reward over Episode")
        self.legend.append('Average Finger Tip Reward')
        self.ax.legend(self.legend)
        # self.ax.set_ylim([-0.61, 0.001])
        self.ax.grid(True)
        self.ax.set_aspect('auto',adjustable='box')
        self.curr_graph = 'Group_Reward'
        return return_finger_rewards
            
    def draw_scatter_end_dist(self, folder_path, cmap='plasma'):
        
        episode_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith('.pkl')]
        filenames_only = [f for f in os.listdir(folder_path) if f.lower().endswith('.pkl')]
        
        filenums = [re.findall('\d+',f) for f in filenames_only]
        final_filenums = []
        for i in filenums:
            if len(i) > 0 :
                final_filenums.append(int(i[0]))
        
        sorted_inds = np.argsort(final_filenums)
        final_filenums = np.array(final_filenums)
        episode_files = np.array(episode_files)
        filenames_only = np.array(filenames_only)
        episode_files = episode_files[sorted_inds].tolist()
        goals, end_dists = [],[]
        for episode_file in episode_files:
            with open(episode_file, 'rb') as ef:
                tempdata = pkl.load(ef)

            data = tempdata['timestep_list']
            try:
                goals.append(data[0]['state']['goal_pose']['goal_position'][0:2])
                if all(np.isclose(data[0]['state']['goal_pose']['goal_position'][0:2],[0.0009830552164485, -0.0687461950930642])):
                    print(episode_file, data[0]['state']['goal_pose']['goal_position'][0:2])
            except KeyError:
                goals.append(data[0]['state']['goal_pose']['goal_pose'][0:2])
                if all(np.isclose(data[0]['state']['goal_pose']['goal_pose'][0:2],[0.0009830552164485, -0.0687461950930642])):
                    print(episode_file, data[0]['state']['goal_pose']['goal_pose'][0:2])
            end_dists.append(data[-1]['reward']['distance_to_goal'])
        # print(end_dists)
        self.clear_axes()
        # linea = np.array([[0.0,0.06],[0.0,-0.06]])*100
        # lineb = np.array([[0.0424,-0.0424],[-0.0424,0.0424]])*100
        # linec = np.array([[0.0424,0.0424],[-0.0424,-0.0424]])*100
        # lined = np.array([[0.06,0.0],[-0.06,0.0]])*100
        goals = np.array(goals)
        # print(goals)
        end_dists = np.array(end_dists)
        mean, std = np.average(end_dists), np.std(end_dists)
        end_dists = np.clip(end_dists, 0, 0.025)

        num_success = end_dists < 0.005
        try:
            a = self.ax.scatter(goals[:,0]*100, goals[:,1]*100, c = end_dists*100, cmap=cmap)
        except:
            a = self.ax.scatter(goals[:,0]*100, goals[:,1]*100, c = end_dists*100, cmap='plasma')
        # self.legend.extend(['Ending Goal Distance'])
        # self.ax.legend(self.legend)
        # self.ax.plot(linea[:,0],linea[:,1])
        # self.ax.plot(lineb[:,0],lineb[:,1])
        # self.ax.plot(linec[:,0],linec[:,1])
        # self.ax.plot(lined[:,0],lined[:,1])
        

        self.ax.set_ylabel('Y position (cm)')
        self.ax.set_xlabel('X position (cm)')
        self.ax.set_xlim([-8,8])
        self.ax.set_ylim([-8,8])
        self.ax.set_title('Distance to Goals')
        self.ax.grid(False)
        self.colorbar = self.fig.colorbar(a, ax=self.ax, extend='max')
        self.ax.set_aspect('equal',adjustable='box')
        self.curr_graph = 'scatter'
        print(f'average end distance {mean} +/- {std}')
        print(f'success percent = {np.sum(num_success)/len(num_success)*100}')
        return [mean, std]
        
    def draw_scatter_contact_dist(self,folder_path):
        
        episode_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith('.pkl')]
        filenames_only = [f for f in os.listdir(folder_path) if f.lower().endswith('.pkl')]
        
        filenums = [re.findall('\d+',f) for f in filenames_only]
        final_filenums = []
        for i in filenums:
            if len(i) > 0 :
                final_filenums.append(int(i[0]))
        
        
        sorted_inds = np.argsort(final_filenums)
        final_filenums = np.array(final_filenums)
        episode_files = np.array(episode_files)
        filenames_only = np.array(filenames_only)
        episode_files = episode_files[sorted_inds].tolist()
        finger_max_dists, timesteps_until_lost_contact, goal_positions = [],[],[]
        
        for episode_file in episode_files:
            with open(episode_file, 'rb') as ef:
                tempdata = pkl.load(ef)

            data = tempdata['timestep_list']
            f1_dists = [f['reward']['f1_dist'] for f in data]
            f2_dists = [f['reward']['f2_dist'] for f in data]
            goal_position = data[0]['state']['goal_pose']['goal_position'][0:2]
            lost_contact = 151
            max_dists = max(max(f1_dists), max(f2_dists))
            for f1,f2,i in zip(f1_dists,f2_dists,range(151)):
                if max([f1,f2]) > 0.001:
                    lost_contact = i
                    break
            finger_max_dists.append(max_dists)
            timesteps_until_lost_contact.append(lost_contact)
            goal_positions.append(goal_position)
        self.clear_axes()
         
        
        goals = np.array(goal_positions)
        finger_max_dists = np.array(finger_max_dists)
        a = self.ax.scatter(goals[:,0]*100, goals[:,1]*100, c = finger_max_dists*100, cmap='jet')
        self.ax.set_ylabel('Y position (cm)')
        self.ax.set_xlabel('X position (cm)')
        self.ax.set_xlim([-7,7])
        self.ax.set_ylim([-7,7])
        self.ax.set_title('Maximum Finger Distance')
        self.ax.grid(False)
        self.colorbar = self.fig.colorbar(a, ax=self.ax)
        self.ax.set_aspect('equal',adjustable='box')
        
        self.curr_graph = 'scatter'
    
    def draw_multifigure_rewards(self,data_dict):

        episode_number = data_dict['number']

        data = data_dict['timestep_list']
        reward_containers = [f['reward'] for f in data]
        
        overall_reward = []
        f1_reward = []
        f2_reward = []
        dist_reward = []
        for reward in reward_containers:
            overall_reward.append(self.build_reward(reward, self.tholds)[0])
            f1_reward.append(reward['f1_dist'])
            f2_reward.append(reward['f2_dist'])
            dist_reward.append(reward['distance_to_goal'])
        
        f1_reward = np.array(f1_reward)
        f2_reward = np.array(f2_reward)
        dist_reward = np.array(dist_reward)
        
        min_dists = np.min([f1_reward,f2_reward,dist_reward])
        max_dists = np.max([f1_reward,f2_reward,dist_reward])
        
        data = data_dict['timestep_list']
        trajectory_points = [f['state']['obj_2']['pose'][0] for f in data]
        fingertip1_points = [f['state']['f1_pos'] for f in data]
        fingertip2_points = [f['state']['f2_pos'] for f in data]
        goal_pose = data[1]['reward']['goal_position']
        trajectory_points = np.array(trajectory_points)
        fingertip1_points = np.array(fingertip1_points)
        fingertip2_points = np.array(fingertip2_points)
        arrow_len = max(int(len(trajectory_points)/25),1)
        arrow_points = np.linspace(0,len(trajectory_points)-arrow_len-1,10,dtype=int)
        next_points = arrow_points + arrow_len
        
        fig, axes = plt.subplots(2,2)

        
        axes[0,1].plot(trajectory_points[:,0], trajectory_points[:,1])
        axes[0,1].plot(fingertip1_points[:,0], fingertip1_points[:,1])
        axes[0,1].plot(fingertip2_points[:,0], fingertip2_points[:,1])
        axes[0,1].plot([trajectory_points[0,0], goal_pose[0]],[trajectory_points[0,1],goal_pose[1]])
        axes[0,1].set_xlim([-0.07,0.07])
        axes[0,1].set_ylim([0.04,0.16])
        axes[0,1].set_xlabel('X pos (m)')
        axes[0,1].set_ylabel('Y pos (m)')
        
        axes[1,1].plot(range(len(f1_reward)), f1_reward)
        axes[1,1].plot(range(len(f1_reward)), f2_reward)
        axes[1,1].set_ylim([min_dists-0.001, max_dists+0.001])
        axes[1,0].plot(range(len(f1_reward)), dist_reward)
        axes[1,0].set_ylim([min_dists-0.001, max_dists+0.001])
        axes[0,0].plot(range(len(f1_reward)), overall_reward)
        axes[1,1].set_title('Contact Distances')
        axes[1,0].set_title('Object Goal Distance')
        
        axes[0,0].set_title('Total Reward')
        
        axes[0,0].grid(True)
        axes[1,0].grid(True)
        axes[1,1].grid(True)
        for i,j in zip(arrow_points,next_points):
            axes[0,1].arrow(trajectory_points[i,0],trajectory_points[i,1], 
                          trajectory_points[j,0]-trajectory_points[i,0],
                          trajectory_points[j,1]-trajectory_points[i,1], 
                          color='blue', width=0.001, head_width = 0.002, length_includes_head=True)
            axes[0,1].arrow(fingertip1_points[i,0],fingertip1_points[i,1], 
                          fingertip1_points[j,0]-fingertip1_points[i,0],
                          fingertip1_points[j,1]-fingertip1_points[i,1],
                          color='orange', width=0.001, head_width = 0.002, length_includes_head=True)
            axes[0,1].arrow(fingertip2_points[i,0],fingertip2_points[i,1], 
                          fingertip2_points[j,0]-fingertip2_points[i,0],
                          fingertip2_points[j,1]-fingertip2_points[i,1],
                          color='green', width=0.001, head_width = 0.002, length_includes_head=True)
            
        # self.legend.extend(['Object Trajectory','Right Finger Trajectory',
        #                     'Left Finger Trajectory','Ideal Path to Goal'])
        # axes[0,1].legend(self.legend)
        axes[0,1].set_title('Object and Finger Path - Episode: '+str(episode_number))
        plt.tight_layout()
        #  
        self.curr_graph = 'path'

        return fig, axes
            
    def draw_end_poses(self, clicks):
        print('buckle up') 
        
        # fig, (ax1,ax2) = plt.subplots(2,1,height_ratios=[2,1])

        fig = plt.figure(constrained_layout=True, figsize=(8,6))
        ax = fig.add_gridspec(4, 3)
        ax1 = fig.add_subplot(ax[0:3, :])
        ax1.set_aspect('equal',adjustable='box')
        ax2 = fig.add_subplot(ax[-1, :])

        self.clear_axes()

        if self.click_spell is None:
            # Rounded Start X      Rounded Start Y
            point_dist = np.sqrt((clicks[0]/100 - self.point_dictionary['Rounded Start X'])**2 + (clicks[1]/100 - self.point_dictionary['Rounded Start Y'])**2)
            mask = np.isclose(point_dist,np.min(point_dist))
            self.click_spell = mask

        end_dists = self.point_dictionary[self.click_spell]['End Distance']
        end_x = self.point_dictionary[self.click_spell]['End X']
        end_y = self.point_dictionary[self.click_spell]['End Y']

        bins = np.linspace(0,0.05,100) + 0.05/100
        num_things = np.zeros(100)
        small_thold = max(0.005,min(end_dists))
        med_thold = small_thold+0.005
        big_thold = med_thold + 0.01 
        end_poses = np.array([end_x,end_y]).transpose()
        small_pose, med_pose, large_pose, fucked = [],[],[],[]
        for pose,dist in zip(end_poses,end_dists):
            if dist <= small_thold:
                small_pose.append(pose)
            elif dist <= med_thold:
                med_pose.append(pose)
            elif dist <= big_thold:
                large_pose.append(pose)
            else:
                fucked.append(pose)
            a= np.where(dist<bins)
            try:
                num_things[a[0][0]] +=1
            except IndexError:
                print('super far away point')
                num_things[-1] +=1
        print(med_pose)
        # ax1.scatter(goals[:,0]*100, goals[:,1]*100)
        ax1.scatter(self.point_dictionary[self.click_spell]['Start X']*100,self.point_dictionary[self.click_spell]['Start Y']*100,marker='s')
        self.legend.append('goal poses')
        if len(fucked)>0:
            
            fucked = np.array(fucked)
            ax1.scatter(fucked[:,0]*100, fucked[:,1]*100)
            self.legend.extend(['> '+str(big_thold*100)+' cm'])
        if len(large_pose)>0:
            large_pose = np.array(large_pose)
            ax1.scatter(large_pose[:,0]*100, large_pose[:,1]*100)
            self.legend.extend(['<= '+str(big_thold*100)+' cm'])
        if len(med_pose)>0:
            med_pose = np.array(med_pose)
            print(med_pose*100)
            ax1.scatter(med_pose[:,0]*100, med_pose[:,1]*100)
            self.legend.extend(['<= '+str(med_thold*100)+' cm'])
        if len(small_pose)>0:
            small_pose = np.array(small_pose)
            ax1.scatter(small_pose[:,0]*100, small_pose[:,1]*100)
            self.legend.extend(['<= '+str(small_thold*100)+' cm'])
        # ax2.bar(['<0.5 cm','<1 cm','<2 cm','>2 cm'], [len(small_pose),len(med_pose),len(large_pose), len(fucked)])
        ax2.bar(bins*100, num_things, width=5/100)
        plt.tight_layout()
        # self.legend.extend(['Ending Goal Distance'])
        # self.ax.legend(self.legend)
        ax1.set_ylabel('Y position (cm)')
        ax1.set_xlabel('X position (cm)')
        ax1.set_xlim([-7,7])
        ax1.set_ylim([-7,7])
        ax1.set_title('Distance to Goals')
        ax1.grid(False)

        ax1.legend(self.legend)
        self.ax.set_aspect('equal',adjustable='box')
        self.curr_graph = 'scatter'
        return fig, (ax1, ax2)
    
    def draw_aout_comparison(self, data_dict):
        episode_number = data_dict['number']
        data = data_dict['timestep_list']
        fingertip1_points = [f['state']['f1_pos'] for f in data]
        fingertip2_points = [f['state']['f2_pos'] for f in data]
        finger1_diff = [[fingertip1_points[i+1][0] - fingertip1_points[i][0],fingertip1_points[i+1][1] - fingertip1_points[i][1]]  for i in range(len(fingertip1_points)-1)]
        finger2_diff = [[fingertip2_points[i+1][0] - fingertip2_points[i][0],fingertip2_points[i+1][1] - fingertip2_points[i][1]]  for i in range(len(fingertip2_points)-1)]

        current_angle_dict = [f['state']['two_finger_gripper']['joint_angles'] for f in data]
        current_angle_list = []
        prev_angles = None
        for angle in current_angle_dict:
            temp = [angs for angs in angle.values()]
            if prev_angles is not None:
                current_angle_list.append(-prev_angles+np.array(temp))        
            prev_angles = np.array(temp)
        current_angle_list = np.array(current_angle_list)/0.1
        finger1_diff = np.array(finger1_diff)/0.01
        finger2_diff = np.array(finger2_diff)/0.01
        # print(finger1_diff,finger2_diff,current_angle_list)
        
        if self.clear_plots | (self.curr_graph !='aout'):
            self.clear_axes()
            
        self.ax.plot(range(len(current_angle_list)),current_angle_list[:,0])
        self.ax.plot(range(len(current_angle_list)),current_angle_list[:,1])
        self.ax.plot(range(len(current_angle_list)),current_angle_list[:,2])
        self.ax.plot(range(len(current_angle_list)),current_angle_list[:,3])
        self.ax.plot(range(len(finger1_diff)),finger1_diff[:,0])
        self.ax.plot(range(len(finger1_diff)),finger1_diff[:,1])
        self.ax.plot(range(len(finger2_diff)),finger2_diff[:,0])
        self.ax.plot(range(len(finger2_diff)),finger2_diff[:,1])
        self.legend.extend(['Right Proximal - episode '+str( episode_number), 
                            'Right Distal - episode '+str( episode_number), 
                            'Left Proximal - episode '+str( episode_number), 
                            'Left Distal - episode '+str( episode_number)])
        self.legend.extend(['Right X - episode ' + str( episode_number), 
                            'Right Y - episode ' + str( episode_number), 
                            'Left X - episode ' + str( episode_number), 
                            'Left Y - episode ' + str( episode_number)])
        self.curr_graph = 'aout'
        self.ax.legend(self.legend)
        self.ax.grid(True)
        self.ax.set_aspect('auto',adjustable='box')
        
    def clear_axes(self):
        if self.colorbar:
            self.colorbar.remove()  
            self.colorbar = None
        self.ax.cla()
        self.legend = []

    def get_figure(self):
        return self.fig, self.ax
    
    def draw_avg_num_goals(self, folder_or_data_dict):
        episode_files = [os.path.join(folder_or_data_dict, f) for f in os.listdir(folder_or_data_dict) if f.lower().endswith('.pkl')]
        filenames_only = [f for f in os.listdir(folder_or_data_dict) if f.lower().endswith('.pkl')]
        
        filenums = [re.findall('\d+',f) for f in filenames_only]
        final_filenums = []
        for i in filenums:
            if len(i) > 0 :
                final_filenums.append(int(i[0]))
        
        sorted_inds = np.argsort(final_filenums)
        final_filenums = np.array(final_filenums)
        temp = final_filenums[sorted_inds]
        episode_files = np.array(episode_files)
        filenames_only = np.array(filenames_only)

        episode_files = episode_files[sorted_inds].tolist()
        
        num_goals = []
        for i, episode_file in enumerate(episode_files):
            with open(episode_file, 'rb') as ef:
                tempdata = pkl.load(ef)
            data = tempdata['timestep_list']
            goal_poses = len(np.unique([i['state']['goal_pose']['goal_position'] for i in data], axis=0))
            num_goals.append(goal_poses)
            if i % 100 ==0:
                print('count = ',i)
        moveing_avg = 100
        num_goals = moving_average(num_goals,moveing_avg)

        self.ax.plot(range(len(num_goals)),num_goals)
        self.ax.set_xlabel('Episode_num')
        self.ax.set_ylabel('Avg goal num')                                                                                                                                                                                                                                   
        self.legend.extend(['Goals'])
        self.ax.legend(self.legend)
        self.ax.set_title('Average goals in 100 episodes')
        self.curr_graph = 'path'
        self.ax.set_aspect('auto',adjustable='box')
    
    def draw_average_efficiency(self, folder_or_data_dict):
        episode_files = [os.path.join(folder_or_data_dict, f) for f in os.listdir(folder_or_data_dict) if f.lower().endswith('.pkl')]
        filenames_only = [f for f in os.listdir(folder_or_data_dict) if f.lower().endswith('.pkl')]
        
        filenums = [re.findall('\d+',f) for f in filenames_only]
        final_filenums = []
        for i in filenums:
            if len(i) > 0 :
                final_filenums.append(int(i[0]))
        
        sorted_inds = np.argsort(final_filenums)
        final_filenums = np.array(final_filenums)
        temp = final_filenums[sorted_inds]
        episode_files = np.array(episode_files)
        filenames_only = np.array(filenames_only)
        count = 0
        episode_files = episode_files[sorted_inds].tolist()
        ending_dists = []
        efficiency=[]
        for episode_file in episode_files:
            with open(episode_file, 'rb') as ef:
                tempdata = pkl.load(ef)

            data = tempdata['timestep_list']
            ending_dists.append(data[-1]['reward']['distance_to_goal'])
            poses = np.array([i['state']['obj_2']['pose'][0][0:2] for i in data])
            end_pos = poses[-1].copy()
            goal_pose = data[-1]['state']['goal_pose']['goal_position'][0:2]
            end_pos[1] -= 0.1
            dist_along_vec = np.dot(end_pos,np.array(goal_pose)/np.linalg.norm(goal_pose))
            dtemp = dist_along_vec
            dist_along_vec = dist_along_vec - abs(np.linalg.norm(goal_pose)-dist_along_vec)
            dist_traveled = [poses[i+1]-poses[i] for i in range(len(poses)-1)]
            temp = [np.linalg.norm(d) for d in dist_traveled]
            mag_dist = np.sum(temp)
            efficiency.append(dist_along_vec/mag_dist)
            if count% 100 ==0:
                print('count = ', count)
            count +=1

        moveing_avg = 100
        mean, std = np.average(efficiency), np.std(efficiency)
        print(f'average efficiency {mean} +/- {std}')
        efficiency = moving_average(efficiency,moveing_avg)
        self.ax.plot(range(len(efficiency)),efficiency)
        self.ax.set_xlabel('Episode_num')
        self.ax.set_ylabel('Avg Movement Efficiency')       
        self.ax.set_ylim(-0.01,1.01)                                                                                                                                                                                                                            
        # self.legend.extend(['Goals'])
        # self.ax.legend(self.legend)
        self.ax.set_aspect('auto',adjustable='box')
        self.ax.set_title('Average Movement Efficiency')
        self.curr_graph = 'path'
        return [mean, std]

    def draw_end_pos_no_color(self,folder_or_data_dict):
        episode_files = [os.path.join(folder_or_data_dict, f) for f in os.listdir(folder_or_data_dict) if f.lower().endswith('.pkl')]
        filenames_only = [f for f in os.listdir(folder_or_data_dict) if f.lower().endswith('.pkl')]
        
        filenums = [re.findall('\d+',f) for f in filenames_only]
        final_filenums = []
        for i in filenums:
            if len(i) > 0 :
                final_filenums.append(int(i[0]))
        
        sorted_inds = np.argsort(final_filenums)
        final_filenums = np.array(final_filenums)
        temp = final_filenums[sorted_inds]
        episode_files = np.array(episode_files)
        filenames_only = np.array(filenames_only)
        count = 0
        episode_files = episode_files[sorted_inds].tolist()
        end_poses = []
        goal_poses=[]
        for episode_file in episode_files:
            with open(episode_file, 'rb') as ef:
                tempdata = pkl.load(ef)

            data = tempdata['timestep_list']
            
            end_poses.append(data[-1]['state']['obj_2']['pose'][0][0:2])
            goal_poses.append(data[-1]['state']['goal_pose']['goal_position'])
            if count% 100 ==0:
                print('count = ', count)
            count +=1
        end_poses = np.array(end_poses)
        goal_poses = np.array(goal_poses)
        self.ax.scatter(goal_poses[:,0], goal_poses[:,1]+0.1)
        self.ax.scatter(end_poses[:,0],end_poses[:,1])

        self.ax.set_xlabel('X position (m)')
        self.ax.set_ylabel('y position (m)')

        self.ax.set_title('Just the things')
        self.ax.legend(['Goal Poses', 'End Poses'])
        self.ax.set_xlim([-0.1,0.1])
        self.ax.set_ylim([0.0,0.2])
        self.ax.set_aspect('equal',adjustable='box')

    def draw_radar(self,folder_list,legend_thing):
        # folder list should be 3 asterisk test folders from the same configuration but different random seeds
        episode_files = []
        for folder_or_data_dict in folder_list:
            ef = [os.path.join(folder_or_data_dict, f) for f in os.listdir(folder_or_data_dict) if (f.lower().endswith('.pkl') and not('2v2' in f))]
            episode_files.extend(ef)
        # print(episode_files)
        end_poses = []
        goal_poses = []
        name_key_og = np.array([[-0.06,0],[-0.0424,0.0424],[0.0,0.06],[0.0424,0.0424],[0.06,0.0],[0.0424,-0.0424],[0.0,-0.06],[-0.0424,-0.0424]])
        name_key = np.array([[0,0.07],[0.0495,0.0495],[0.07,0.0],[0.0495,-0.0495],[0.0,-0.07],[-0.0495,-0.0495],[-0.07,0.0],[-0.0495,0.0495]])
        name_key2 = ["N","NE","E","SE","S","SW", "W","NW"]
        name_key_og2 = ["E","NE","N","NW","W","SW","S","SE"]
        dist_traveled_list = []
        for episode_file in episode_files:
            with open(episode_file, 'rb') as ef:
                tempdata = pkl.load(ef)
            # print(tempdata)
            if type(tempdata) is dict:
                data = tempdata['timestep_list']
            else:
                data = tempdata

            poses = np.array([i['state']['obj_2']['pose'][0][0:2] for i in data])
            dist_traveled = [poses[i+1]-poses[i] for i in range(len(poses)-1)]
            temp = [np.linalg.norm(d) for d in dist_traveled]
            mag_dist = np.sum(temp)
            dist_traveled_list.append(mag_dist)
            end_poses.append(data[-1]['state']['obj_2']['pose'][0][0:2])
            goal_poses.append(data[-1]['state']['goal_pose']['goal_position'])
            # if count% 100 ==0:
            #     print('count = ', count)
            # count +=1
        end_poses = np.array(end_poses)
        end_poses = end_poses - np.array([0,0.1])
        dist_along_thing = {'E':[],'NE':[],'N':[],'NW':[],'W':[],'SW':[],'S':[],'SE':[]}
        efficiency = {'E':[],'NE':[],'N':[],'NW':[],'W':[],'SW':[],'S':[],'SE':[]}
        # goal_poses = np.array
        for e, g, dt in zip(end_poses, goal_poses, dist_traveled_list):
            for i,name in enumerate(name_key):
                # print(name,g)
                if all(name == g):
                    # print('we in here')
                    dtemp = g/np.linalg.norm(g)*np.dot(e,g/np.linalg.norm(g))
                    dist_along_thing[name_key2[i]].append(dtemp)
                    efficiency[name_key2[i]].append(np.linalg.norm(dtemp)/dt)
            for i,name in enumerate(name_key_og):
                # print(name,g)
                if all(name == g):
                    dtemp = g/np.linalg.norm(g)*np.dot(e,g/np.linalg.norm(g))
                    dist_along_thing[name_key_og2[i]].append(dtemp)
                    efficiency[name_key_og2[i]].append(np.linalg.norm(dtemp)/dt)
        # print(dist_along_thing)
        # print('efficiency', efficiency, dist_traveled_list)
        # print(np.unique(goal_poses,axis=0))
        finals = []
        alls = []
        net_efficiency = []
        for k in name_key2:
            # print(k, dist_along_thing[k])
            finals.append(np.average(dist_along_thing[k],axis=0))
            alls.append(np.linalg.norm(dist_along_thing[k][0]))
            net_efficiency.append(efficiency[k][0])
            try:
                alls.append(np.linalg.norm(dist_along_thing[k][1]))
                net_efficiency.append(efficiency[k][1])
                alls.append(np.linalg.norm(dist_along_thing[k][2]))
                net_efficiency.append(efficiency[k][2])
            except:
                pass
        finals.append(finals[0])
        finals = np.array(finals)
        print(legend_thing)
        print(f'net efficiency: {np.average(net_efficiency)}, {np.std(net_efficiency)}')
        # print('total distance from the avg',np.sum(np.linalg.norm(finals[0:8],axis=1)))
        print(f'what we need. mean: {np.average(alls)*8}, {np.std(alls)}')
        # print()
        self.ax.plot(finals[:,0],finals[:,1]+0.1)
        # self.ax.fill(finals[:,0],finals[:,1]+0.1, alpha=0.3)
        self.ax.set_xlim([-0.08,0.08])
        self.ax.set_ylim([0.04,0.16])
        self.ax.set_xlabel('X pos (m)')
        self.ax.set_ylabel('Y pos (m)')
        self.legend.append(legend_thing)
        self.ax.legend(self.legend)
        self.ax.set_aspect('equal',adjustable='box')
        # self.ax.scatter(end_poses[:,0],end_poses[:,1])
        return [np.average(alls)*8, np.std(alls), np.average(net_efficiency), np.std(net_efficiency)]

    def draw_rotation_asterisk(self,folder_list,legend_thing):
        '''
        Ill be honest i dont know how we are going to plot this shit but we will find out together
        Right now the main thing is to be able to get the averages
        Once again dealing with two sources of error is a challenge
        We expect folder list to have all the damn folders
        '''
        print(legend_thing)
        episode_files = []
        for folder_or_data_dict in folder_list:
            ef = [os.path.join(folder_or_data_dict, f) for f in os.listdir(folder_or_data_dict) if (f.lower().endswith('.pkl') and not('2v2' in f))]
            episode_files.extend(ef)
        # print(episode_files)
        end_poses = []
        end_orientations = []
        goal_poses = []
        goal_orientations = []
        pos_name_key = ['bottom','center','left','right','top']
        dir_name_key = ['/Clockwise','/CounterClockwise']
        full_key = []
        sim_key = ['/Episode_5','/Episode_1','/Episode_9','/Episode_7','/Episode_3','/Episode_4','/Episode_0','/Episode_8','/Episode_6','/Episode_2']
        for i in pos_name_key:
            for j in dir_name_key:
                full_key.append('_'.join([j,i]))
        dist_traveled_list = []
        names=[]
        for episode_file in episode_files:
            with open(episode_file, 'rb') as ef:
                tempdata = pkl.load(ef)
            # print(tempdata)
            if type(tempdata) is dict:
                data = tempdata['timestep_list']
            else:
                data = tempdata
            poses = np.array([i['state']['obj_2']['pose'][0][0:2] for i in data])
            orientations = np.array([i['state']['obj_2']['pose'][1] for i in data])
            goal_o = data[0]['state']['goal_pose']['goal_orientation']
            end_o = R.from_quat(orientations[-1])
            end_o = end_o.as_euler('xyz')[-1]
            
            dist_traveled = [poses[i+1]-poses[i] for i in range(len(poses)-1)]
            temp = [np.linalg.norm(d) for d in dist_traveled]
            mag_dist = np.sum(temp)
            dist_traveled_list.append(mag_dist)
            end_poses.append(data[-1]['state']['obj_2']['pose'][0][0:2])
            goal_poses.append(data[-1]['state']['goal_pose']['goal_position'])
            end_orientations.append(end_o)
            goal_orientations.append(goal_o)
            names.append(episode_file)
            # print(end_poses, goal_poses, end_orientations, goal_orientations)
            # if count% 100 ==0:
            #     print('count = ', count)
            # count +=1
        end_poses = np.array(end_poses)
        end_poses = end_poses - np.array([0,0.1])
        pos_error = {'/Clockwise_center':[],'/Clockwise_bottom':[],'/Clockwise_top':[],'/Clockwise_left':[],'/Clockwise_right':[],
                     '/CounterClockwise_center':[],'/CounterClockwise_bottom':[],'/CounterClockwise_top':[],'/CounterClockwise_left':[],'/CounterClockwise_right':[]}
        orientation_error = {'/Clockwise_center':[],'/Clockwise_bottom':[],'/Clockwise_top':[],'/Clockwise_left':[],'/Clockwise_right':[],
                     '/CounterClockwise_center':[],'/CounterClockwise_bottom':[],'/CounterClockwise_top':[],'/CounterClockwise_left':[],'/CounterClockwise_right':[]}        
        
        orientation_covered = {'/Clockwise_center':[],'/Clockwise_bottom':[],'/Clockwise_top':[],'/Clockwise_left':[],'/Clockwise_right':[],
                     '/CounterClockwise_center':[],'/CounterClockwise_bottom':[],'/CounterClockwise_top':[],'/CounterClockwise_left':[],'/CounterClockwise_right':[]}
                # goal_poses = np.array
        all_pos_error = []
        all_orientation_error=[]
        all_or_covered = []
        for g, gp, ep, go, eo in zip(names,goal_poses, end_poses, goal_orientations, end_orientations):
            print(g)
            for i,name in enumerate(full_key):
                if name in g:
                    # print('we in here')
                    dtemp = ep-gp
                    pos_error[full_key[i]].append(abs(dtemp)*100)
                    orientation_error[full_key[i]].append(abs(go-eo)*180/np.pi)
                    orientation_covered[full_key[i]].append(np.sign(eo)*go*180/np.pi) 
                    all_orientation_error.append(abs(go-eo)*180/np.pi)
                    all_pos_error.append(abs(dtemp)*100)
                    all_or_covered.append(np.sign(eo)*go*180/np.pi)
                    # print('starting debugging')
                    # print(dtemp)
                    # print(go-eo)
                    print(np.sign(go)*eo)
            for i, name in enumerate(sim_key):
                # print(name,g)
                if name in g:
                    print('we in here')
                    dtemp = ep-gp
                    pos_error[full_key[i]].append(abs(dtemp)*100)
                    orientation_error[full_key[i]].append(abs(go-eo)*180/np.pi)
                    orientation_covered[full_key[i]].append(np.sign(eo)*go*180/np.pi)
                    all_orientation_error.append(abs(go-eo)*180/np.pi)
                    all_pos_error.append(abs(dtemp)*100)
                    all_or_covered.append(np.sign(eo)*go*180/np.pi)
        # finals.append(finals[0])
        # finals = np.array(finals)
        print('net orientation covered', np.average(all_or_covered))
        # print(f'net position error: {np.average(all_pos_error)}, {np.std(all_pos_error)}')
        # print(f'net orientation error: {np.average(all_orientation_error)}, {np.std(all_orientation_error)}')
        # print(f'avereage orientation traveled: {np.average(all_or_covered)}, {np.std(all_or_covered)}')
        # print('total distance from the avg',np.sum(np.linalg.norm(finals[0:8],axis=1)))

        # print()
        # self.ax.plot(finals[:,0],finals[:,1]+0.1)
        # # self.ax.fill(finals[:,0],finals[:,1]+0.1, alpha=0.3)
        # self.ax.set_xlim([-0.08,0.08])
        # self.ax.set_ylim([0.04,0.16])
        # self.ax.set_xlabel('X pos (m)')
        # self.ax.set_ylabel('Y pos (m)')
        # self.legend.append(legend_thing)
        # self.ax.legend(self.legend)
        # self.ax.set_aspect('equal',adjustable='box')
        return [np.average(all_pos_error), np.std(all_pos_error), np.average(all_orientation_error), np.std(all_orientation_error), np.average(all_or_covered),np.std(all_or_covered)]

    def draw_orientation_success_rate(self,folder):
        starttime = time.time()
        self.clear_axes()
        rotations = []
        goals = []
        episode_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith('.pkl')]
        filenames_only = [f for f in os.listdir(folder) if f.lower().endswith('.pkl')]
        
        filenums = [re.findall('\d+',f) for f in filenames_only]
        final_filenums = []
        for i in filenums:
            if len(i) > 0 :
                final_filenums.append(int(i[0]))
        
        sorted_inds = np.argsort(final_filenums)
        final_filenums = np.array(final_filenums)
        episode_files = np.array(episode_files)
        filenames_only = np.array(filenames_only)

        episode_files = episode_files[sorted_inds].tolist()
        fail_rewards = []
        fail_rotation = [] 
        success_rewards = []
        success_rotation = [] 
        count = 0
        pool = multiprocessing.Pool()
        tst = (-1, -1, -1)
        keys = (('state','obj_2','z_angle'),('state','goal_pose','goal_orientation'),('reward','distance_to_goal'))
        thing = [[ef, tst, keys] for ef in episode_files]
        print('applying async')
        data_list = pool.starmap(pool_key_list,thing)
        pool.close()
        pool.join()
        distances = []
        for i in data_list:
            if i[2]<0.026:
                success_rewards.append(i[0])
                success_rotation.append(i[1])
            else:    
                fail_rewards.append(i[0])
                fail_rotation.append(i[1])
        
        success_rewards = np.array(success_rewards)*180/np.pi
        success_rotation = np.array(success_rotation) *180/np.pi
        fail_rewards = np.array(fail_rewards)*180/np.pi
        fail_rotation = np.array(fail_rotation) *180/np.pi

        # a = np.abs(rewards)
        # b = np.average(a)
        # c = np.std(a)
        # print('average and std orientation error', b,c)
        self.ax.scatter(success_rotation,success_rewards)
        self.ax.scatter(fail_rotation,fail_rewards)
        self.ax.plot([-360,360],[-360,360],color='green')

        self.ax.set_xlabel('Goal Orientation')
        self.ax.set_ylabel('Ending Orientation')
        self.ax.set_ylim(-75,75)
        self.ax.set_xlim(-75,75)
        self.ax.set_aspect('equal',adjustable='box')
        self.ax.legend(['Angles within Threshold','Angles outside Threshold','Ideal Behavior'])

    def draw_orientation(self,data_dict):
        self.clear_axes()
        data = data_dict['timestep_list']
        rotations = []
        goals = []
        for tstep in data:
            obj_rotation = tstep['reward']['object_orientation'][2]
            obj_rotation = (obj_rotation + np.pi)%(np.pi*2)
            obj_rotation = (obj_rotation - np.pi)*180/np.pi
            rotations.append(obj_rotation)
            goals.append(tstep['reward']['goal_orientation']*180/np.pi)
        print(data[0]['reward'])
        print(data[0]['state'])
        self.ax.plot(range(len(rotations)), rotations)
        self.ax.plot(range(len(goals)), goals)
        self.ax.set_xlabel('timestep')
        self.ax.set_ylabel('angle (deg)')
        self.ax.set_aspect('auto',adjustable='box')
        self.ax.legend(['Object Angle','Goal Angle'])
        
    def draw_finger_goal_path(self, data_dict):
        try:
            data = data_dict['timestep_list']
            episode_number = data_dict['number']
            trajectory_points = [f['state']['obj_2']['pose'][0] for f in data]
            fingertip1_points = [f['state']['f1_pos'] for f in data]
            fingertip2_points = [f['state']['f2_pos'] for f in data]
            goal_pose = data[1]['reward']['goal_position']
            trajectory_points = np.array(trajectory_points)
            fingertip1_points = np.array(fingertip1_points)
            fingertip2_points = np.array(fingertip2_points)
            arrow_len = max(int(len(trajectory_points)/25),1)
            arrow_points = np.linspace(0,len(trajectory_points)-arrow_len-1,10,dtype=int)
            next_points = arrow_points + arrow_len
            goal_points = data[0]['reward']['goal_finger']
            t = goal_points[0]
            self.clear_axes()
            
            self.ax.plot(trajectory_points[:,0], trajectory_points[:,1])
            self.ax.plot(fingertip1_points[:,0], fingertip1_points[:,1])
            self.ax.plot(fingertip2_points[:,0], fingertip2_points[:,1])
            self.ax.plot([trajectory_points[0,0], goal_pose[0]],[trajectory_points[0,1],goal_pose[1]])
            self.ax.plot(fingertip1_points[0,0], fingertip1_points[0,1], marker='o',
                        markersize=5,markerfacecolor='orange',markeredgecolor='orange')
            self.ax.plot(fingertip2_points[0,0], fingertip2_points[0,1], marker='o',
                        markersize=5,markerfacecolor='green',markeredgecolor='green')
            self.ax.plot(goal_points[0], goal_points[1], marker='o',
                        markersize=5,markerfacecolor='orange',markeredgecolor='orange')
            self.ax.plot(goal_points[2], goal_points[3], marker='o',
                        markersize=5,markerfacecolor='green',markeredgecolor='green')

            self.ax.set_xlim([-0.08,0.08])
            self.ax.set_ylim([0.02,0.18])
            self.ax.set_xlabel('X pos (m)')
            self.ax.set_ylabel('Y pos (m)')
            for i,j in zip(arrow_points,next_points):
                self.ax.arrow(trajectory_points[i,0],trajectory_points[i,1], 
                            trajectory_points[j,0]-trajectory_points[i,0],
                            trajectory_points[j,1]-trajectory_points[i,1], 
                            color='blue', width=0.001, head_width = 0.002, length_includes_head=True)
                self.ax.arrow(fingertip1_points[i,0],fingertip1_points[i,1], 
                            fingertip1_points[j,0]-fingertip1_points[i,0],
                            fingertip1_points[j,1]-fingertip1_points[i,1],
                            color='orange', width=0.001, head_width = 0.002, length_includes_head=True)
                self.ax.arrow(fingertip2_points[i,0],fingertip2_points[i,1], 
                            fingertip2_points[j,0]-fingertip2_points[i,0],
                            fingertip2_points[j,1]-fingertip2_points[i,1],
                            color='green', width=0.001, head_width = 0.002, length_includes_head=True)
                
            self.legend.extend(['Object Trajectory','Right Finger Trajectory',
                                'Left Finger Trajectory','Ideal Path to Goal'])
            self.ax.legend(self.legend)
            self.ax.set_title('Object and Finger Path - Episode: '+str(episode_number))
            
            self.curr_graph = 'path'
        except TypeError:
            print('Finger goals are None, this was not from a finger goal task.')
        
    def draw_relative_reward_strength(self,folder,tholds):        
        # get list of pkl files in folder
        episode_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith('.pkl')]
        filenames_only = [f for f in os.listdir(folder) if f.lower().endswith('.pkl')]
        
        filenums = [re.findall('\d+',f) for f in filenames_only]
        final_filenums = []
        for i in filenums:
            if len(i) > 0 :
                final_filenums.append(int(i[0]))
        
        
        sorted_inds = np.argsort(final_filenums)
        final_filenums = np.array(final_filenums)
        temp = final_filenums[sorted_inds]
        episode_files = np.array(episode_files)
        filenames_only = np.array(filenames_only)

        episode_files = episode_files[sorted_inds].tolist()

        datapoints = [] 
        pool = multiprocessing.Pool()
        keys = (('reward','distance_to_goal'), ('reward', 'object_orientation'), ('reward','goal_orientation'), ('reward', 'f1_dist'), ('reward','f2_dist'))
        thing = [[ef, keys] for ef in episode_files]
        print('applying async')
        data_list = pool.starmap(pool_key_list_all,thing)
        pool.close()
        pool.join()


        sliding_rewards = []
        contact_rewards = []
        orientation_rewards = []
        for i in data_list:
            sliding_rewards.append(-np.sum(i[0])/0.01*tholds['DISTANCE_SCALING'])
            contact_rewards.append(-np.sum(np.max([np.array(i[3]),np.array(i[4])],axis=0))*100*tholds['CONTACT_SCALING'])
            orientation_rewards.append(-np.sum(np.abs(np.array(i[1])[:,2]-np.array(i[2])))*tholds['ROTATION_SCALING'])
        
        sliding_rewards = np.array(sliding_rewards)
        contact_rewards = np.array(contact_rewards)
        orientation_rewards = np.array(orientation_rewards)
        rewards = sliding_rewards + contact_rewards + orientation_rewards

        # print(rewards)
        return_rewards = rewards.copy()
        if self.moving_avg != 1:
            contact_rewards = moving_average(contact_rewards,self.moving_avg)
            orientation_rewards = moving_average(orientation_rewards,self.moving_avg)
            sliding_rewards = moving_average(sliding_rewards,self.moving_avg)
            rewards = moving_average(rewards,self.moving_avg)
        if self.clear_plots | (self.curr_graph !='Group_Reward'):
            self.clear_axes()
             
        self.legend.append('Average Distance Reward')
        self.ax.plot(range(len(sliding_rewards)), sliding_rewards)
        self.ax.plot(range(len(orientation_rewards)), orientation_rewards)
        self.ax.plot(range(len(contact_rewards)), contact_rewards)
        self.ax.plot(range(len(rewards)),rewards)
        self.ax.set_xlabel('Episode Number')
        self.ax.set_ylabel('Total Reward Over the Entire Episode')
        self.ax.set_title("Agent Reward over Episode")
        # self.ax.set_ylim([-12, 0])
        self.ax.legend(['Sliding Rewards','Orientation Rewards','Contact Rewards','Net Rewards'])
        self.ax.grid(True)
        self.ax.set_aspect('auto',adjustable='box')
        self.curr_graph = 'Group_Reward'
        return return_rewards
        
    def draw_rotation_sliding_error(self,folder,cmap):
        self.clear_axes()

        episode_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith('.pkl')]
        filenames_only = [f for f in os.listdir(folder) if f.lower().endswith('.pkl')]
        
        filenums = [re.findall('\d+',f) for f in filenames_only]
        final_filenums = []
        for i in filenums:
            if len(i) > 0 :
                final_filenums.append(int(i[0]))
        
        sorted_inds = np.argsort(final_filenums)
        final_filenums = np.array(final_filenums)
        episode_files = np.array(episode_files)
        filenames_only = np.array(filenames_only)

        episode_files = episode_files[sorted_inds].tolist()

        pool = multiprocessing.Pool()
        keys = (('state','goal_pose','goal_orientation'),('state','obj_2','z_angle'),('reward','distance_to_goal'))
        tst = (-1,-1,-1)
        thing = [[ef, tst, keys] for ef in episode_files]
        print('applying async')
        data_list = pool.starmap(pool_key_list,thing)
        pool.close()
        pool.join()
        rewards = []
        rotation = []  
        goal_dist = []
        for i in data_list:
            rewards.append(i[0]-i[1])
            rotation.append(i[0])
            goal_dist.append(i[2])

        rewards = np.array(rewards) * 180/np.pi
        rotation = np.array(rotation) * 180/np.pi
        goal_dist = np.array(goal_dist)
        # print(rewards,rotation)
        rewards = np.clip(np.abs(rewards),0,30)
        a=self.ax.scatter(rotation,goal_dist,c = np.abs(rewards), cmap=cmap)
        # self.ax.plot(range(len(goals)), goals)
        # self.ax.plot([-360,360],[-360,360],color='orange')
        self.ax.set_xlabel('Goal Orientation')
        self.ax.set_ylabel('Ending Slide Error')
        self.colorbar = self.fig.colorbar(a, ax=self.ax, extend='max')
        self.ax.set_title('Goal Orientation Error')
        self.ax.set_ylim(-0.001,0.15)
        # self.ax.set_xlim(-95,95)
        self.ax.set_aspect('auto')
        # self.ax.legend(['Achieved Angles','No Movement Line'])

    def draw_timestep_goal_best(self, folder, tstep):
        # get list of pkl files in folder
        episode_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith('.pkl')]
        filenames_only = [f for f in os.listdir(folder) if f.lower().endswith('.pkl')]
        
        filenums = [re.findall('\d+',f) for f in filenames_only]
        final_filenums = []
        for i in filenums:
            if len(i) > 0 :
                final_filenums.append(int(i[0]))
        
        sorted_inds = np.argsort(final_filenums)
        final_filenums = np.array(final_filenums)
        temp = final_filenums[sorted_inds]
        episode_files = np.array(episode_files)
        filenames_only = np.array(filenames_only)

        episode_files = episode_files[sorted_inds].tolist()
        thing = [[i,tstep] for i in episode_files]
        print('applying async', tstep, len(episode_files))
        pool = multiprocessing.Pool()
        min_dists = pool.starmap(goal_dist_process,thing)
        pool.close()
        pool.join()

        return_mins = min_dists.copy()
        if self.moving_avg != 1:
            min_dists = moving_average(min_dists,self.moving_avg)
        if self.clear_plots | (self.curr_graph != 'goal_dist'):
            self.clear_axes()
             
        self.ax.plot(range(len(min_dists)),min_dists)
        self.legend.extend([f'Best Goal Distance at step {tstep}'])
        self.ax.legend(self.legend)
        self.ax.set_ylabel('Goal Distance')
        self.ax.set_xlabel('Episode')
        self.ax.grid(True)
        self.ax.set_title('Distance to Goal Per Episode')
        self.ax.set_aspect('auto',adjustable='box')
        self.curr_graph = 'goal_dist'
        print('average and std dev of positions', np.average(return_mins), np.std(return_mins))
        return return_mins
    
    def draw_timestep_goal_end(self, folder, tstep):
        # get list of pkl files in folder
        episode_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith('.pkl')]
        filenames_only = [f for f in os.listdir(folder) if f.lower().endswith('.pkl')]
        
        filenums = [re.findall('\d+',f) for f in filenames_only]
        final_filenums = []
        for i in filenums:
            if len(i) > 0 :
                final_filenums.append(int(i[0]))
        
        sorted_inds = np.argsort(final_filenums)
        final_filenums = np.array(final_filenums)
        temp = final_filenums[sorted_inds]
        episode_files = np.array(episode_files)
        filenames_only = np.array(filenames_only)

        episode_files = episode_files[sorted_inds].tolist()
        tst = (tstep,)
        keys = (('reward','distance_to_goal'),)
        thing = [[ef, tst, keys] for ef in episode_files]
        print('applying async')

        pool = multiprocessing.Pool()
        data_list = pool.starmap(pool_key_list,thing)
        pool.close()
        pool.join()

        return_mins = data_list.copy()
        if self.moving_avg != 1:
            data_list = moving_average(data_list,self.moving_avg)
        if self.clear_plots | (self.curr_graph != 'goal_dist'):
            self.clear_axes()
             
        self.ax.plot(range(len(data_list)),data_list)
        self.legend.extend([f'Goal Distance at step {tstep}'])
        self.ax.legend(self.legend)
        self.ax.set_ylabel('Goal Distance')
        self.ax.set_xlabel('Episode')
        self.ax.grid(True)
        self.ax.set_title('Distance to Goal Per Episode')
        self.ax.set_aspect('auto',adjustable='box')
        self.curr_graph = 'goal_dist'
        print('average and std dev of positions', np.average(return_mins), np.std(return_mins))
        return return_mins

    def build_scatter_magic(self,folder_path):
        self.point_dictionary = {}
        '''
        This is only here because making it general was a pain in the ass. 
        the x values are unique for the pattern i chose so we can use those to 
        determine which point we are at
        '''

        # num_per_group = 128
        # num_per_group = 1200
        # print(folder_path, type(folder_path))
        if type(folder_path) is str:
            episode_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith('.pkl')]
            filenames_only = [f for f in os.listdir(folder_path) if f.lower().endswith('.pkl')]
            filenums = [re.findall('\d+',f) for f in filenames_only]
            final_filenums = []
            for i in filenums:
                if len(i) > 0 :
                    final_filenums.append(int(i[0]))

            sorted_inds = np.argsort(final_filenums)
            episode_files = np.array(episode_files)
            episode_files = episode_files[sorted_inds].tolist()
        elif type(folder_path) is list:
            episode_files = []
            filenames_only = []
            for path in folder_path:
                ef = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith('.pkl')]
                fo = [f for f in os.listdir(path) if f.lower().endswith('.pkl')]
                filenums = [re.findall('\d+',f) for f in fo]
                final_filenums = []
                for i in filenums:
                    if len(i) > 0 :
                        final_filenums.append(int(i[0]))
                sorted_inds = np.argsort(final_filenums)
                ef = np.array(ef)
                ef = ef[sorted_inds].tolist()
                episode_files.extend(ef)

        print('applying async')
        pool = multiprocessing.Pool()
        data_list = pool.map(pool_process,episode_files)
        pool.close()
        pool.join()
        column_key = ['Start X','Start Y','End X','End Y','Goal X','Goal Y','Start Distance','End Distance', 'Max Distance',
                      'End Orientation','Goal Orientation','Path']
        self.point_dictionary = pd.DataFrame(data_list, columns = column_key)
        self.point_dictionary['Rounded Start X'] = self.point_dictionary['Start X'].apply(lambda x:np.round(x,4))
        self.point_dictionary['Rounded Start Y'] = self.point_dictionary['Start Y'].apply(lambda x:np.round(x,4))
        self.point_dictionary['Orientation Error'] = self.point_dictionary['Goal Orientation'] - self.point_dictionary['End Orientation']
        # print(sys.getsizeof(self.point_dictionary))

    def save_point_dictionary(self, filepath, filename=None):
        if self.point_dictionary is None:
            print('no point dictionary yet dingus')
            return
        else:
            if filename is None:
                self.point_dictionary.to_pickle(filepath+'/combined_data.pkl')
            else:
                self.point_dictionary.to_pickle(filepath+'/'+filename+'.pkl')

    def load_point_dictionary(self,picklename):
        self.point_dictionary = pd.read_pickle(picklename)

    def draw_scatter_end_magic(self, folder_path, cmap='plasma'):
        if self.point_dictionary is None:
            self.build_scatter_magic(folder_path)

        self.clear_axes()
        mean=np.average(self.point_dictionary['End Distance'])
        std = np.std(self.point_dictionary['End Distance'])
        sorted = self.point_dictionary.groupby(['Rounded Start X','Rounded Start Y'])

        end_dists = sorted['End Distance'].apply(np.average)
        start_x = sorted['Start X'].apply(np.average)
        start_y = sorted['Start Y'].apply(np.average)
        end_dists = np.clip(end_dists, 0, 0.025)

        try:
            a = self.ax.scatter(start_x.to_numpy()*100, start_y.to_numpy()*100, c = end_dists*100, cmap=cmap,vmin=0.0, vmax=2.5)
        except:
            a = self.ax.scatter(start_x.to_numpy()*100, start_y.to_numpy()*100, c = end_dists*100, cmap='plasma',vmin=0.0, vmax=2.5)

        self.ax.set_ylabel('Y position (cm)')
        self.ax.set_xlabel('X position (cm)')
        self.ax.set_xlim([-8,8])
        self.ax.set_ylim([-8,8])
        self.ax.set_title('Distance to Goals')
        self.ax.grid(False)
        self.colorbar = self.fig.colorbar(a, ax=self.ax, extend='max')
        # self.colorbar.set_clim(vmin=0.0, vmax=2.5)
        self.ax.set_aspect('equal',adjustable='box')
        self.curr_graph = 'scatter'
        print(f'average end distance {mean} +/- {std}')
        self.click_spell = None
        return [mean, std]

    def draw_scatter_spell(self, clicks, cmap='plasma'):
        if self.point_dictionary is None:
            print('cant do it, need to run wizard first')
            return
        if clicks[0] is None:
            print('need to select a point first')
            return

        self.clear_axes()

        if self.click_spell is None:
            # Rounded Start X      Rounded Start Y
            point_dist = np.sqrt((clicks[0]/100 - self.point_dictionary['Rounded Start X'])**2 + (clicks[1]/100 - self.point_dictionary['Rounded Start Y'])**2)
            mask = np.isclose(point_dist,np.min(point_dist))
            self.click_spell = mask

        end_dists = self.point_dictionary[self.click_spell]['End Distance']
        goal_x = self.point_dictionary[self.click_spell]['Goal X']
        goal_y = self.point_dictionary[self.click_spell]['Goal Y']

        mean, std = np.average(end_dists), np.std(end_dists)
        end_dists = np.clip(end_dists, 0, 0.025)
        
        try:
            a = self.ax.scatter(goal_x*100, goal_y*100, c = end_dists*100, cmap=cmap)
        except:
            a = self.ax.scatter(goal_x*100, goal_y*100, c = end_dists*100, cmap='plasma')

        self.ax.set_ylabel('Y position (cm)')
        self.ax.set_xlabel('X position (cm)')
        self.ax.set_xlim([-8,8])
        self.ax.set_ylim([-8,8])
        self.ax.set_title('Distance to Goals')
        self.ax.grid(False)
        self.colorbar = self.fig.colorbar(a, ax=self.ax, extend='max')
        self.ax.scatter(self.point_dictionary[self.click_spell]['Start X']*100,self.point_dictionary[self.click_spell]['Start Y']*100,marker='s')
        self.ax.set_aspect('equal',adjustable='box')
        self.curr_graph = 'scatter'
        print(f'average end distance {mean} +/- {std}')
        return [mean, std]
    
    def draw_dist_relationship(self,clicks, cmap='plasma'):
        # get list of pkl files in folder
        if self.point_dictionary is None:
            print('cant do it, need to run wizard first')
            return
        if clicks[0] is None:
            print('need to select a point first')
            return

        closest_point = []
        distances = []
        test_point = np.array(clicks)/100
        desired_point = []
        start_pos, end_dists, goal_points = [], [], []
        # print(type(self.point_dictionary))
        for k,v in self.point_dictionary.items():
            if len(v['goal_pos']) > 0:
                ind = np.argmin(np.linalg.norm(test_point-np.array(v['goal_pos']),axis=1))
                # print(ind, len(v['start_pos']), len(v['dist']), len(v['goal_pos']))
                start_pos.append(v['start_pos'][ind])
                end_dists.append(v['dist'][ind])
                goal_points.append(v['goal_pos'][ind])

        end_dists = np.array(end_dists)
        mean, std = np.average(end_dists), np.std(end_dists)
        end_dists = np.clip(end_dists, 0, 0.025)
        self.click_spell = None
        self.clear_axes()
        start_pos = np.array(start_pos)
        end_dists = np.array(end_dists)
        end_dists= np.clip(end_dists, 0, 0.025)
        a = self.ax.scatter(start_pos[:,0]*100, start_pos[:,1]*100, c = end_dists*100, cmap=cmap)
        self.ax.scatter(goal_points[0][0]*100,goal_points[0][1]*100,marker='s')
        self.ax.set_ylabel('Y position (cm)')
        self.ax.set_xlabel('X position (cm)')
        self.ax.set_xlim([-8,8])
        self.ax.set_ylim([-8,8])
        self.ax.set_title('Ending Distance Based on START Position')
        self.ax.grid(False)
        self.colorbar = self.fig.colorbar(a, ax=self.ax, extend='max')
        self.ax.set_aspect('equal',adjustable='box')
        
        self.curr_graph = 'scatter'

    def draw_path_spell(self, clicks):
        if self.point_dictionary is None:
            print('cant do it, need to run wizard first')
            return False
        if self.click_spell is None:
            print('cant do it, need to run scatter spell first')
            return False
        
        point_dist = np.sqrt((clicks[0]/100 - self.point_dictionary[self.click_spell]['Goal X'])**2 + (clicks[1]/100 - self.point_dictionary[self.click_spell]['Goal Y'])**2)
        specific_value = np.argmin(point_dist)
        point = self.point_dictionary[self.click_spell].loc[self.point_dictionary[self.click_spell]['Path'].keys()[specific_value]]
        datapath = point['Path']
        with open(datapath, 'rb') as file:
            data_dict = pkl.load(file)
        data = data_dict['timestep_list']
        episode_number=data_dict['number']
        trajectory_points = [f['state']['obj_2']['pose'][0] for f in data]
        print('goal position in state', data[0]['state']['goal_pose'])
        try:
            goal_poses = np.array([i['state']['goal_pose']['goal_pose'] for i in data])
        except:
            goal_poses = np.array([i['state']['goal_pose']['goal_position'] for i in data])
        # print(trajectory_points)
        trajectory_points = np.array(trajectory_points)
        ideal = np.zeros([len(goal_poses)+1,2])
        ideal[0,:] = trajectory_points[0,0:2]
        ideal[1:,:] = goal_poses + np.array([0,0.1])
        if self.clear_plots | (self.curr_graph != 'path'):
            self.clear_axes()
        self.ax.plot(trajectory_points[:,0], trajectory_points[:,1])
        self.ax.plot(ideal[:,0],ideal[:,1])
        self.ax.set_xlim([-0.08,0.08])
        self.ax.set_ylim([0.02,0.18])
        self.ax.set_xlabel('X pos (m)')
        self.ax.set_ylabel('Y pos (m)')                                                                                                                                                                                                                                   
        self.legend.extend(['RL Object Trajectory - episode '+str(episode_number), 'Ideal Path to Goal - episode '+str(episode_number)])
        self.ax.legend(self.legend)
        self.ax.set_title('Object Path')
        self.ax.set_aspect('equal',adjustable='box')
        self.curr_graph = 'path'
        # print(data[0]['state']['direction'])
        filename = datapath.split('/')[-1]
        return filename

    def draw_orientation_end_magic(self, folder_path, cmap='plasma'):
        if self.point_dictionary is None:
            self.build_scatter_magic(folder_path)

        self.clear_axes()
        mean=np.average(np.abs(self.point_dictionary['Orientation Error']))*180/np.pi
        std = np.std(np.abs(self.point_dictionary['Orientation Error'])) * 180/np.pi
        sorted = self.point_dictionary.groupby(['Rounded Start X','Rounded Start Y'])

        end_orientations = sorted['Orientation Error'].apply(lambda x:np.average(np.abs(x)))
        start_x = sorted['Start X'].apply(np.average)
        start_y = sorted['Start Y'].apply(np.average)

        try:
            a = self.ax.scatter(start_x.to_numpy()*100, start_y.to_numpy()*100, c = end_orientations*180/np.pi, cmap=cmap,vmin=0.0, vmax=15)
        except:
            a = self.ax.scatter(start_x.to_numpy()*100, start_y.to_numpy()*100, c = end_orientations*180/np.pi, cmap='plasma',vmin=0.0, vmax=15)

        self.ax.set_ylabel('Y position (cm)')
        self.ax.set_xlabel('X position (cm)')
        self.ax.set_xlim([-8,8])
        self.ax.set_ylim([-8,8])
        self.ax.set_title('Average Orientation Error (degrees) Based on Start Pose')
        self.ax.grid(False)
        self.colorbar = self.fig.colorbar(a, ax=self.ax, extend='max')
        # self.colorbar.set_clim()
        self.ax.set_aspect('equal',adjustable='box')
        self.curr_graph = 'scatter'
        print(f'average end orientation {mean} +/- {std}')
        self.click_spell = None
        return [mean, std]

    def draw_scatter_orientation_spell(self, clicks, cmap='plasma'):
        if self.point_dictionary is None:
            print('cant do it, need to run wizard first')
            return
        if clicks[0] is None:
            print('need to select a point first')
            return

        self.clear_axes()

        if self.click_spell is None:
            # Rounded Start X      Rounded Start Y
            point_dist = np.sqrt((clicks[0]/100 - self.point_dictionary['Rounded Start X'])**2 + (clicks[1]/100 - self.point_dictionary['Rounded Start Y'])**2)
            mask = np.isclose(point_dist,np.min(point_dist))
            self.click_spell = mask
        
        orienatation_errors = []
        goals = []
        end_orientation = self.point_dictionary[self.click_spell]['End Orientation']
        goal_orientation = self.point_dictionary[self.click_spell]['Goal Orientation']
        goal_x = self.point_dictionary[self.click_spell]['Goal X']
        goal_y = self.point_dictionary[self.click_spell]['Goal Y']

        orienatation_errors = np.abs(end_orientation-goal_orientation)*180/np.pi
        mean, std = np.average(orienatation_errors), np.std(orienatation_errors)
        orienatation_errors = np.clip(orienatation_errors,0,15)
        goals = np.array(goals)
        try:
            a = self.ax.scatter(goal_x*100, goal_y*100, c = orienatation_errors, cmap=cmap)
        except:
            a = self.ax.scatter(goal_x*100, goal_y*100, c = orienatation_errors, cmap='plasma')

        self.ax.set_ylabel('Y position (cm)')
        self.ax.set_xlabel('X position (cm)')
        self.ax.set_xlim([-8,8])
        self.ax.set_ylim([-8,8])
        self.ax.set_title('Orientation (deg) Based on Goal Pose')
        self.ax.grid(False)
        self.colorbar = self.fig.colorbar(a, ax=self.ax, extend='max')
        self.ax.set_aspect('equal',adjustable='box')
        self.curr_graph = 'scatter'
        print(f'average orientation error {mean} +/- {std}')
        return [mean, std]

    def draw_orientation_spell(self, clicks):
        if self.point_dictionary is None:
            print('cant do it, need to run wizard first')
            return False
        if self.click_spell is None:
            print('cant do it, need to run scatter spell first')
            return False
        
        # Rounded Start X      Rounded Start Y
        point_dist = np.sqrt((clicks[0]/100 - self.point_dictionary[self.click_spell]['Goal X'])**2 + (clicks[1]/100 - self.point_dictionary[self.click_spell]['Goal Y'])**2)
        specific_value = np.argmin(point_dist)
        point = self.point_dictionary[self.click_spell].loc[self.point_dictionary[self.click_spell]['Path'].keys()[specific_value]]
        datapath = point['Path']
        with open(datapath, 'rb') as file:
            data_dict = pkl.load(file)
        self.clear_axes()
        data = data_dict['timestep_list']
        rotations = []
        goals = []
        for tstep in data:
            obj_rotation = tstep['reward']['object_orientation'][2]
            obj_rotation = (obj_rotation + np.pi)%(np.pi*2)
            obj_rotation = (obj_rotation - np.pi)*180/np.pi
            rotations.append(obj_rotation)
            goals.append(tstep['reward']['goal_orientation']*180/np.pi)

        self.ax.plot(range(len(rotations)), rotations)
        self.ax.plot(range(len(goals)), goals)
        self.ax.set_xlabel('timestep')
        self.ax.set_ylabel('angle (deg)')
        self.ax.set_aspect('auto',adjustable='box')
        self.ax.legend(['Object Angle','Goal Angle'])
        filename = datapath.split('/')[-1]
        return filename
    
    def draw_success_high_level(self,folder_path,tholds):
        if self.point_dictionary is None:
            self.build_scatter_magic(folder_path)

        self.clear_axes()
        mean=np.average(np.abs(self.point_dictionary['Orientation Error']))*180/np.pi
        std = np.std(np.abs(self.point_dictionary['Orientation Error'])) * 180/np.pi
        sorted = self.point_dictionary.groupby(['Rounded Start X','Rounded Start Y'])

        start_x = sorted['Start X'].apply(np.average)*100
        start_y = sorted['Start Y'].apply(np.average)*100
        def s_f_func(group):
            return np.average((abs(group['Orientation Error'])*180/np.pi < tholds[1]) & (group['End Distance'] < tholds[0]/1000))
        s_f = sorted.apply(s_f_func)
        a=self.ax.scatter(start_x, start_y, c = s_f, cmap='plasma')
        self.ax.set_ylabel('Y position (cm)')
        self.ax.set_xlabel('X position (cm)')
        self.ax.set_xlim([-8,8])
        self.ax.set_ylim([-8,8])
        self.ax.set_title('Success Rate Plot')
        self.ax.grid(False)
        self.colorbar = self.fig.colorbar(a, ax=self.ax, label='Success Rate', extend='max')
        self.colorbar.mappable.set_clim(0.0,1.0)
        self.click_spell = None
    
    def draw_success_scatter(self, clicks, success_range, rot_success):
        if self.point_dictionary is None:
            print('cant do it, need to run wizard first')
            return
        if clicks[0] is None:
            print('need to select a point first')
            return

        self.clear_axes()

        if self.click_spell is None:
            # Rounded Start X      Rounded Start Y
            point_dist = np.sqrt((clicks[0]/100 - self.point_dictionary['Rounded Start X'])**2 + (clicks[1]/100 - self.point_dictionary['Rounded Start Y'])**2)
            mask = np.isclose(point_dist,np.min(point_dist))
            self.click_spell = mask
        goal_x = self.point_dictionary[self.click_spell]['Goal X']
        goal_y = self.point_dictionary[self.click_spell]['Goal Y']
        def s_f_func(pt):
            # print(pt.keys())
            return (abs(pt['Orientation Error'])*180/np.pi < rot_success) & (pt['End Distance'] < success_range/1000)
        s_f = self.point_dictionary[self.click_spell].apply(s_f_func, axis=1)
        try:
            a = self.ax.scatter(goal_x*100, goal_y*100, c = s_f)
        except:
            a = self.ax.scatter(goal_x*100, goal_y*100, c = s_f)

        self.ax.set_ylabel('Y position (cm)')
        self.ax.set_xlabel('X position (cm)')
        self.ax.set_xlim([-8,8])
        self.ax.set_ylim([-8,8])
        self.ax.set_title('Distance to Goals')
        self.ax.grid(False)
        self.colorbar = self.fig.colorbar(a, ax=self.ax, extend='max')
        # self.ax.scatter(self.point_dictionary[self.click_spell]['start_pos'][0][0]*100,self.point_dictionary[self.click_spell]['start_pos'][0][1]*100,marker='s')
        self.ax.set_aspect('equal',adjustable='box')
        self.curr_graph = 'scatter'
        # print(f'average end distance {mean} +/- {std}')
        # return [mean, std]
 
    def draw_end_pose_shenanigans(self,folder_path, cmap='plasma'):
        print(folder_path)
        if self.point_dictionary is None:
            self.build_scatter_magic(folder_path)

        self.clear_axes()

        mean=np.average(self.point_dictionary['End Distance'])
        std = np.std(self.point_dictionary['End Distance'])
        sorted = self.point_dictionary.groupby(['Goal X','Goal Y'])

        end_dists = sorted['End Distance'].apply(np.average)
        goal_x = sorted['Goal X'].apply(np.average)
        goal_y = sorted['Goal Y'].apply(np.average)
        # end_dists = np.clip(end_dists, 0, 0.025)

        try:
            a = self.ax.scatter(goal_x.to_numpy()*100, goal_y.to_numpy()*100, c = end_dists*100, cmap=cmap)
        except:
            a = self.ax.scatter(goal_x.to_numpy()*100, goal_y.to_numpy()*100, c = end_dists*100, cmap='plasma')

        self.ax.set_ylabel('Y position (cm)')
        self.ax.set_xlabel('X position (cm)')
        self.ax.set_xlim([-8,8])
        self.ax.set_ylim([-8,8])
        self.ax.set_title('Distance to Goals')
        self.ax.grid(False)
        self.colorbar = self.fig.colorbar(a, ax=self.ax, extend='max')
        self.ax.set_aspect('equal',adjustable='box')
        self.curr_graph = 'scatter'
        self.colorbar.mappable.set_clim(0.3,2.5)
        print(f'average end distance {mean} +/- {std}')
        self.click_spell = None
        return [mean, std]
    
    def draw_scatter_scaled_dist(self, folder_path, cmap='plasma'):
        if self.point_dictionary is None:
            self.build_scatter_magic(folder_path)

        self.clear_axes()
        end_poses = []
        distances = []
        points_for_std = []
        points = np.zeros((1200,37))
        point_key = []
        count=0
        for _,v in self.point_dictionary.items():
            if len(v['goal_pos']) > 0:
                if len(point_key) == 0:
                    point_key = v['goal_pos']
                    point_key = np.array(point_key)
                for point, dist,start in zip(v['goal_pos'], v['dist'], v['start_dist']):
                    ind = np.argwhere(point_key==point)[0][0]
                    points[ind,count] = 1-dist/start
            count +=1
        end_dists = np.average(points,axis=1)
        points_for_std = points.copy()
        print(np.average(points))
        print(points)
        end_dists = np.clip(end_dists, 0, 1)
        
        goals = point_key
        try:
            a = self.ax.scatter(goals[:,0]*100, goals[:,1]*100, c = end_dists, cmap=cmap)
        except:
            a = self.ax.scatter(goals[:,0]*100, goals[:,1]*100, c = end_dists, cmap='plasma')

        mean, std = np.average(points_for_std), np.std(points_for_std)

        self.ax.set_ylabel('Y position (cm)')
        self.ax.set_xlabel('X position (cm)')
        self.ax.set_xlim([-8,8])
        self.ax.set_ylim([-8,8])
        self.ax.set_title('Average Percent of Point Reached')
        self.ax.grid(False)
        self.colorbar = self.fig.colorbar(a, ax=self.ax, extend='max')
        self.ax.set_aspect('equal',adjustable='box')
        self.curr_graph = 'scatter'
        print(f'average end distance {mean} +/- {std}')
        self.click_spell = None
        return [mean, std]
    
    def draw_orientation_region(self,clicks, slide_thold, rotate_thold):
        if self.point_dictionary is None:
            print('cant do it, need to run wizard first')
            return False
        if clicks[0] is None:
            print('need to select a point first')
            return
        self.clear_axes()

        if self.click_spell is None:
            # Rounded Start X      Rounded Start Y
            point_dist = np.sqrt((clicks[0]/100 - self.point_dictionary['Rounded Start X'])**2 + (clicks[1]/100 - self.point_dictionary['Rounded Start Y'])**2)
            mask = np.isclose(point_dist,np.min(point_dist))
            self.click_spell = mask

        success_end_orientation = self.point_dictionary[(self.click_spell) & (self.point_dictionary['End Distance'] < slide_thold/1000)]['End Orientation']
        success_goal_orientation = self.point_dictionary[(self.click_spell) & (self.point_dictionary['End Distance'] < slide_thold/1000)]['Goal Orientation']

        fail_end_orientation = self.point_dictionary[(self.click_spell) & (self.point_dictionary['End Distance'] >= slide_thold/1000)]['End Orientation']
        fail_goal_orientation = self.point_dictionary[(self.click_spell) & (self.point_dictionary['End Distance'] >= slide_thold/1000)]['Goal Orientation']


        fail_end_orientation = np.array(fail_end_orientation)*180/np.pi
        fail_goal_orientation = np.array(fail_goal_orientation)*180/np.pi
        success_end_orientation = np.array(success_end_orientation)*180/np.pi
        success_goal_orientation = np.array(success_goal_orientation)*180/np.pi

        self.ax.scatter(success_goal_orientation,success_end_orientation)
        self.ax.scatter(fail_goal_orientation,fail_end_orientation)
        # self.ax.plot(range(len(goals)), goals)
        self.ax.plot([-360,360],[-360,360],color='green')
        self.ax.plot([-360,360],[-360+rotate_thold,360+rotate_thold],color='green',linestyle=':')
        self.ax.plot([-360,360],[-360-rotate_thold,360-rotate_thold],color='green',linestyle=':')
        self.ax.set_xlabel('Goal Orientation')
        self.ax.set_ylabel('Ending Orientation Error')
        self.ax.set_ylim(-75,75)
        self.ax.set_xlim(-75,75)
        self.ax.set_aspect('equal',adjustable='box')
        self.ax.legend(['Achieved Angles within Threshold','Achieved Angles outside Threshold','Ideal Behavior'])

    def draw_end_orientaion_buckets(self, folder_path):
        print('buckle up') 
        if self.point_dictionary is None:
            self.build_scatter_magic(folder_path)

        fig = plt.figure(constrained_layout=True, figsize=(8,6))
        ax = fig.add_gridspec(4, 3)
        ax1 = fig.add_subplot(ax[0:2, :])
        ax2 = fig.add_subplot(ax[2:3, :])
        ax3 = fig.add_subplot(ax[:-1, :])

        self.clear_axes()

        end_dists = self.point_dictionary['End Distance']
        end_orientation = self.point_dictionary['End Orientation']
        goal_orientation = self.point_dictionary['Goal Orientation']
        orientation_error = np.abs(end_orientation - goal_orientation) * 180/np.pi
        translation_bins = np.linspace(0,0.05,100) + 0.05/100
        translation_num_things = np.zeros(100)

        for dist in end_dists:
            a= np.where(dist<translation_bins)
            try:
                translation_num_things[a[0][0]] +=1
            except IndexError:
                print('super far away point')
                translation_num_things[-1] +=1
        orientation_bins = np.linspace(0,90,100)
        orientation_num_things = np.zeros(100)
        for oe in orientation_error:
            a= np.where(oe<orientation_bins)
            try:
                orientation_num_things[a[0][0]] +=1
            except IndexError:
                print('super far away point', oe)
                orientation_num_things[-1] +=1
        # ax1.scatter(goals[:,0]*100, goals[:,1]*100)
        ax1.bar(orientation_bins,orientation_num_things, width=90/100)
        ax1.set_xlabel('Orientation Error (deg)')
        ax1.set_ylabel('Number of Trials')
        ax2.bar(translation_bins*100, translation_num_things, width=5/100)
        ax2.set_xlabel('Translational Error (cm)')
        ax2.set_ylabel('Number of Trials')
        self.ax.set_aspect('equal',adjustable='box')
        self.curr_graph = 'scatter'
        return fig, (ax1, ax2)
    
    def draw_both_errors(self, folder_path):
        if self.point_dictionary is None:
            self.build_scatter_magic(folder_path)

        self.clear_axes()

        sorted = self.point_dictionary.sort_values(by=['Orientation Error'])
        x = abs(sorted['Orientation Error']) * 180/np.pi
        y = sorted['End Distance'] * 100
        self.ax.scatter(x,y,s=2)
        self.ax.set_xlabel('Orientation Error (deg)')
        self.ax.set_ylabel('Distance Error (cm)')
        self.ax.set_aspect('auto')

    def draw_scatter_max_end(self,folder_path, cmap):
        if self.point_dictionary is None:
            self.build_scatter_magic(folder_path)

        self.clear_axes()
        distances = self.point_dictionary['Max Distance']- self.point_dictionary['End Distance']
        mean=np.average(distances)
        std = np.std(distances)

        def subtract_and_average(group):
            return np.average(group['Max Distance'] - group['End Distance'])
        sorted = self.point_dictionary.groupby(['Rounded Start X','Rounded Start Y'])

        end_dists = sorted.apply(subtract_and_average)
        start_x = sorted['Start X'].apply(np.average)
        start_y = sorted['Start Y'].apply(np.average)
        # end_dists = np.clip(end_dists, 0, 0.025)

        try:
            a = self.ax.scatter(start_x.to_numpy()*100, start_y.to_numpy()*100, c = end_dists*100, cmap=cmap)
        except:
            a = self.ax.scatter(start_x.to_numpy()*100, start_y.to_numpy()*100, c = end_dists*100, cmap='plasma')

        self.ax.set_ylabel('Y position (cm)')
        self.ax.set_xlabel('X position (cm)')
        self.ax.set_xlim([-8,8])
        self.ax.set_ylim([-8,8])
        self.ax.set_title('Max Distance - End Distance')
        self.ax.grid(False)
        self.colorbar = self.fig.colorbar(a, ax=self.ax, extend='max',label='Max Distance - Ending Distance')
        self.ax.set_aspect('equal',adjustable='box')
        self.curr_graph = 'scatter'
        print(f'average end distance {mean} +/- {std}')
        self.click_spell = None
        return [mean, std]

    def draw_newshit(self,clicks, slide_thold):
        if self.point_dictionary is None:
            print('cant do it, need to run wizard first')
            return False
        if clicks[0] is None:
            print('need to select a point first')
            return
        self.clear_axes()

        if self.click_spell is None:
            # Rounded Start X      Rounded Start Y
            point_dist = np.sqrt((clicks[0]/100 - self.point_dictionary['Rounded Start X'])**2 + (clicks[1]/100 - self.point_dictionary['Rounded Start Y'])**2)
            mask = np.isclose(point_dist,np.min(point_dist))
            self.click_spell = mask

        success_end_orientation = self.point_dictionary[(self.click_spell) & (self.point_dictionary['End Distance'] < slide_thold/1000)]['End Orientation']
        success_goal_orientation = self.point_dictionary[(self.click_spell) & (self.point_dictionary['End Distance'] < slide_thold/1000)]['Goal Orientation']

        fail_end_orientation = self.point_dictionary[(self.click_spell) & (self.point_dictionary['End Distance'] >= slide_thold/1000)]['End Orientation']
        fail_goal_orientation = self.point_dictionary[(self.click_spell) & (self.point_dictionary['End Distance'] >= slide_thold/1000)]['Goal Orientation']


        fail_distances = self.point_dictionary[(self.click_spell) & (self.point_dictionary['End Distance'] >= slide_thold/1000)]['Max Distance'] - self.point_dictionary[(self.click_spell) & (self.point_dictionary['End Distance'] >= slide_thold/1000)]['End Distance']
        success_distances = self.point_dictionary[(self.click_spell) & (self.point_dictionary['End Distance'] < slide_thold/1000)]['Max Distance'] - self.point_dictionary[(self.click_spell) & (self.point_dictionary['End Distance'] < slide_thold/1000)]['End Distance']

        fail_end_orientation = np.array(fail_end_orientation)*180/np.pi
        fail_goal_orientation = np.array(fail_goal_orientation)*180/np.pi
        success_end_orientation = np.array(success_end_orientation)*180/np.pi
        success_goal_orientation = np.array(success_goal_orientation)*180/np.pi

        a = self.ax.scatter(success_goal_orientation,success_end_orientation, c=success_distances*100, marker='o', cmap='plasma')
        self.ax.scatter(fail_goal_orientation,fail_end_orientation, c=fail_distances*100, marker='D', cmap='plasma')
        # self.ax.plot(range(len(goals)), goals)
        self.ax.plot([-360,360],[-360,360],color='green')
        self.ax.set_xlabel('Goal Orientation')
        self.ax.set_ylabel('Ending Orientation Error')
        self.ax.set_ylim(-75,75)
        self.ax.set_xlim(-75,75)
        self.ax.set_aspect('equal',adjustable='box')
        self.ax.legend(['Achieved Angles within Threshold','Achieved Angles outside Threshold','Ideal Behavior'])
        self.colorbar = self.fig.colorbar(a, ax=self.ax, extend='max', label='Max Distance - Ending Distance')