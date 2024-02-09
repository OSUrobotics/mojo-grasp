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
import time
from matplotlib.patches import Rectangle
from scipy.spatial.transform import Rotation as R

def moving_average(a, n) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

class PlotBackend():
    def __init__(self, config_folder):
        self.fig, self.ax = plt.subplots()
        self.clear_plots = True
         
        self.curr_graph = None
        self.moving_avg = 1 
        self.colorbar = None
        self.reduced_format = False
        self.config = {}
        self.legend = []
        self.load_config(config_folder)
        
    def load_config(self, config_folder):
        with open(config_folder+'/experiment_config.json') as file:
            self.config = json.load(file)

    def draw_path(self,data_dict):
        data = data_dict['timestep_list']
        episode_number=data_dict['number']
        trajectory_points = [f['state']['obj_2']['pose'][0] for f in data]
        print('goal position in state', data[0]['state']['goal_pose'])
        goal_poses = np.array([i['state']['goal_pose']['goal_pose'] for i in data])
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
        self.curr_graph = 'path'

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
        
    def draw_actor_output(self, data_dict):
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
        if self.config['action'] == 'Finger Tip Position':
            self.legend.extend(['Right X - episode ' + str( episode_number), 
                                'Right Y - episode ' + str( episode_number), 
                                'Left X - episode ' + str( episode_number), 
                                'Left Y - episode ' + str( episode_number)])
        elif self.config['action'] == 'Joint Velocity':            
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

    def draw_distance_rewards(self, data_dict):
        episode_number = data_dict['number']

        data = data_dict['timestep_list']
        current_reward_dict = [-f['reward']['distance_to_goal'] for f in data]

        if self.clear_plots | (self.curr_graph != 'rewards'):
            self.clear_axes()
             
        self.ax.plot(range(len(current_reward_dict)),current_reward_dict)
        self.legend.extend(['Distance Reward - episode ' + str( episode_number)])
        self.ax.legend(self.legend)
        self.ax.grid(True)
        self.ax.set_ylabel('Reward')
        self.ax.set_xlabel('Timestep (1/30 s)')
        self.ax.set_title('Reward Plot')
         
        self.curr_graph = 'rewards'
    
    def draw_contact_rewards(self, data_dict):
        episode_number = data_dict['number']

        data = data_dict['timestep_list']
        current_reward_dict1 = [-f['reward']['f1_dist'] for f in data]
        current_reward_dict2 = [-f['reward']['f2_dist'] for f in data]
        if self.clear_plots | (self.curr_graph != 'rewards'):
            self.clear_axes()
             
        self.ax.plot(range(len(current_reward_dict1)),current_reward_dict1)
        self.ax.plot(range(len(current_reward_dict2)),current_reward_dict2)
        self.legend.extend(['Right Finger Contact Reward - episode ' + str( episode_number),'Left Finger Contact Reward - episode ' + str( episode_number)])
        self.ax.legend(self.legend)
        self.ax.grid(True)
        self.ax.set_ylabel('Reward')
        self.ax.set_xlabel('Timestep (1/30 s)')
        self.ax.set_title('Contact Reward Plot')
         
        self.curr_graph = 'rewards'
        
    def draw_combined_rewards(self, data_dict):
        episode_number = data_dict['number']

        data = data_dict['timestep_list']
        full_reward = []
        general_reward = [f['reward'] for f in data]

        for reward_container in general_reward:
            temp = self.build_reward(reward_container)
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
         
        self.curr_graph = 'rewards'
         
    def draw_explored_region(self, all_data_dict):
        datapoints = []
        for episode in all_data_dict['episode_list']:
            data = episode['timestep_list']
            for timestep in data:
                datapoints.append(timestep['state']['obj_2']['pose'][0][0:2])
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
        ylim = [0.06, 0.26]
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
        
        self.colorbar = self.fig.colorbar(c, ax=self.ax)
         
        self.curr_graph = 'explored'

    def draw_end_region(self, folder_or_data_dict):
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
            datapoints = []
            for episode_file in episode_files:
                with open(episode_file, 'rb') as ef:
                    tempdata = pkl.load(ef)

                data = tempdata['timestep_list']
                datapoints.append(data[-1]['state']['obj_2']['pose'][0][0:2])
                if count% 100 ==0:
                    print('count = ', count)
                count +=1
        elif type(folder_or_data_dict) is dict:
            datapoints = []
            for episode in folder_or_data_dict['episode_list']:
                data = episode['timestep_list']
                datapoints.append(data[-1]['state']['obj_2']['pose'][0][0:2])
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
        ylim = [0.06, 0.26]
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
                    individual_rewards.append(self.build_reward(tstep['reward'])[0])
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
                        individual_rewards.append(self.build_reward(timestep['reward'])[0])
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
            # filenames_only = filenames_only[sorted_inds].tolist()
            rewards = []
            temp = 0
            count = 0
            finger_obj_avgs= np.zeros((len(episode_files),2))
            for i,episode_file in enumerate(episode_files):
                with open(episode_file, 'rb') as ef:
                    tempdata = pkl.load(ef)
                data = tempdata['timestep_list']
                finger_obj_dists = np.zeros((len(data),2))
                for j, timestep in enumerate(data):
                    finger_obj_dists[j,:]=[timestep['reward']['f1_dist'],timestep['reward']['f2_dist']]
                finger_obj_avgs[i] = np.average(finger_obj_dists, axis=0)
                temp = 0
                if count% 100 ==0:
                    print('count = ', count)
                count +=1
        elif type(folder_or_data_dict) is dict:
            finger_obj_avgs = np.zeros((len(folder_or_data_dict['episode_list']),2))
            for i, episode in enumerate(folder_or_data_dict['episode_list']):
                data = episode['timestep_list']
                finger_obj_dists = np.zeros((len(data),2))
                for j, timestep in enumerate(data):
                    finger_obj_dists[j,:]=[timestep['reward']['f1_dist'],timestep['reward']['f2_dist']]
                finger_obj_avgs[i] = np.average(finger_obj_dists, axis=0)
        
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
            # filenames_only = filenames_only[sorted_inds].tolist()
            rewards = []
            temp = 0
            count = 0
            finger_obj_maxs= np.zeros((len(episode_files),2))
            for i,episode_file in enumerate(episode_files):
                with open(episode_file, 'rb') as ef:
                    tempdata = pkl.load(ef)
                data = tempdata['timestep_list']
                finger_obj_dists = np.zeros((len(data),2))
                for j, timestep in enumerate(data):
                    finger_obj_dists[j,:]=[timestep['reward']['f1_dist'],timestep['reward']['f2_dist']]
                finger_obj_maxs[i] = np.max(finger_obj_dists, axis=0)
                temp = 0
                if count% 100 ==0:
                    print('count = ', count)
                count +=1
        elif type(folder_or_data_dict) is dict:
            finger_obj_maxs = np.zeros((len(folder_or_data_dict['episode_list']),2))
            for i, episode in enumerate(folder_or_data_dict['episode_list']):
                data = episode['timestep_list']
                finger_obj_dists = np.zeros((len(data),2))
                for j, timestep in enumerate(data):
                    finger_obj_dists[j,:]=[timestep['reward']['f1_dist'],timestep['reward']['f2_dist']]
                finger_obj_maxs[i] = np.max(finger_obj_dists, axis=0)

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
         
    def draw_avg_actor_output(self, folder_or_data_dict):
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
        if self.config['action'] == 'Finger Tip Position':
            self.legend.extend(['Right X', 
                                'Right Y', 
                                'Left X', 
                                'Left Y'])
        elif self.config['action'] == 'Joint Velocity':            
            self.legend.extend(['Right Proximal', 
                                'Right Distal', 
                                'Left Proximal', 
                                'Left Distal'])
        self.ax.legend(self.legend)
        self.ax.set_ylabel('Actor Output')
        self.ax.set_xlabel('Episode')
        self.ax.grid(True)
        self.ax.set_title('Actor Output')
         
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
         
        self.curr_graph = 'vel'

    def draw_success_rate(self, folder_or_data_dict, success_range):
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
            for episode_file in episode_files:
                with open(episode_file, 'rb') as ef:
                    tempdata = pkl.load(ef)

                data = tempdata['timestep_list']
                ending_dists.append(data[-1]['reward']['distance_to_goal'])
                if count% 100 ==0:
                    print('count = ', count)
                count +=1
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
        for dist in ending_dists:
            if dist < success_range:
                s_f.append(100)
            else:
                s_f.append(0)
        return_dists = ending_dists.copy()

        if self.moving_avg != 1:
            s_f = moving_average(s_f,self.moving_avg)
        if self.clear_plots | (self.curr_graph != 's_f'):
            self.clear_axes()
             
        self.ax.plot(range(len(s_f)),s_f)
        self.legend.extend(['Success Rate (' + str(success_range*1000) + ' mm tolerance)'])
        self.ax.legend(self.legend)
        self.ax.set_ylabel('Success Percentage')
        self.ax.set_xlabel('Episode')
        self.ax.set_ylim([-1,101])
        titlething = 'Percent of Trials over ' + str(self.moving_avg)+' window that are successful'
        self.ax.set_title(titlething)
        self.ax.grid(True)
         
        self.curr_graph = 's_f'
        return return_dists

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
         
        self.curr_graph = 'direction_success_thing' 

    def draw_actor_max_percent(self, folder_path):
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
        
        actor_max = []
        
        for i, episode_file in enumerate(episode_files):
            with open(episode_file, 'rb') as ef:
                tempdata = pkl.load(ef)
            data = tempdata['timestep_list']
            actor_list = [abs(f['action']['actor_output'])>0.99 for f in data]
            actor_list = np.array(actor_list)
            end_actor = np.sum(actor_list,axis=0)/len(actor_list)
            actor_max.append(end_actor)
            if i % 100 ==0:
                print('count = ',i)
        actor_max = np.array(actor_max)
        if self.clear_plots | (self.curr_graph != 'angles'):
            self.clear_axes()
             
        self.ax.plot(range(len(actor_max)),actor_max[:,0])
        self.ax.plot(range(len(actor_max)),actor_max[:,1])
        self.ax.plot(range(len(actor_max)),actor_max[:,2])
        self.ax.plot(range(len(actor_max)),actor_max[:,3])
        if self.config['action'] == 'Finger Tip Position':
            self.legend.extend(['Right X Percent at Max', 
                                'Right Y Percent at Max', 
                                'Left X Percent at Max', 
                                'Left Y Percent at Max'])
        elif self.config['action'] == 'Joint Velocity':            
            self.legend.extend(['Right Proximal Percent at Max', 
                                'Right Distal Percent at Max', 
                                'Left Proximal Percent at Max', 
                                'Left Distal Percent at Max'])
        self.ax.legend(self.legend)
        self.ax.grid(True)
        self.ax.set_ylabel('Actor Output')
        self.ax.set_xlabel('Timestep (1/30 s)')
        self.ax.set_title('Fraction of Episode that Action is Maxed')
         
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
        self.ax.plot([trajectory_points[0,0], goal_pose[0]],[trajectory_points[0,1],goal_pose[1]])
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
            goals.append(data[0]['state']['goal_pose']['goal_pose'][0:2])
            if all(np.isclose(data[0]['state']['goal_pose']['goal_pose'][0:2],[0.0009830552164485, -0.0687461950930642])):
                print(episode_file, data[0]['state']['goal_pose']['goal_pose'][0:2])
            end_dists.append(data[-1]['reward']['distance_to_goal'])
        self.clear_axes()
        # linea = np.array([[0.0,0.06],[0.0,-0.06]])*100
        # lineb = np.array([[0.0424,-0.0424],[-0.0424,0.0424]])*100
        # linec = np.array([[0.0424,0.0424],[-0.0424,-0.0424]])*100
        # lined = np.array([[0.06,0.0],[-0.06,0.0]])*100
        goals = np.array(goals)
        # print(goals)
        end_dists = np.array(end_dists)
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
        mean, std = np.average(end_dists), np.std(end_dists)

        self.ax.set_ylabel('Y position (cm)')
        self.ax.set_xlabel('X position (cm)')
        self.ax.set_xlim([-8,8])
        self.ax.set_ylim([-8,8])
        self.ax.set_title('Distance to Goals')
        self.ax.grid(False)
        self.colorbar = self.fig.colorbar(a, ax=self.ax, extend='max')
         
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
            goal_position = data[0]['state']['goal_pose']['goal_pose'][0:2]
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
         
        self.curr_graph = 'scatter'
    
    def build_reward(self, reward_container):
        """
        Method takes in a Reward object
        Extracts reward information from state_container and returns it as a float
        based on the reward structure contained in self.config['reward']

        :param state: :func:`~mojograsp.simcore.reward.Reward` object.
        :type state: :func:`~mojograsp.simcore.reward.Reward`
        """        
        done2 = False


        if self.config['reward'] == 'Sparse':
            tstep_reward = -1 + 2*(reward_container['distance_to_goal'] < float(self.config['sr'])/1000)
        elif self.config['reward'] == 'Distance':
            tstep_reward = max(-reward_container['distance_to_goal'],-1)
        elif self.config['reward'] == 'Distance + Finger':
            tstep_reward = max(-reward_container['distance_to_goal']*float(self.config['distance_scaling']) - max(reward_container['f1_dist'],reward_container['f2_dist'])*float(self.config['contact_scaling']),-1)
        elif self.config['reward'] == 'Hinge Distance + Finger':
            tstep_reward = reward_container['distance_to_goal'] < float(self.config['sr'])/1000 + max(-reward_container['distance_to_goal'] - max(reward_container['f1_dist'],reward_container['f2_dist'])*float(self.config['contact_scaling']),-1)
        elif self.config['reward'] == 'Slope':
            tstep_reward = reward_container['slope_to_goal'] * float(self.config['distance_scaling'])
        elif self.config['reward'] == 'Slope + Finger':
            tstep_reward = max(reward_container['slope_to_goal'] * float(self.config['distance_scaling'])  - max(reward_container['f1_dist'],reward_container['f2_dist'])*float(self.config['contact_scaling']),-1)
        elif self.config['reward'] == 'SmartDistance + Finger':
            ftemp = max(reward_container['f1_dist'],reward_container['f2_dist'])
            temp = -reward_container['distance_to_goal'] * (1 + 4*reward_container['plane_side'])
            # print(reward_container['plane_side'])
            tstep_reward = max(temp*float(self.config['distance_scaling']) - ftemp*float(self.config['contact_scaling']),-1)
        elif self.config['reward'] == 'ScaledDistance + Finger':
            ftemp = max(reward_container['f1_dist'], reward_container['f2_dist']) * 100 # 100 here to make ftemp = -1 when at 1 cm
            temp = -reward_container['distance_to_goal']/reward_container['start_dist'] * (1 + 4*reward_container['plane_side'])
            # print(reward_container['plane_side'])
            tstep_reward = temp*float(self.config['distance_scaling']) - ftemp*float(self.config['contact_scaling'])
        elif self.config['reward'] == 'ScaledDistance+ScaledFinger':
            ftemp = -max(reward_container['f1_dist'], reward_container['f2_dist']) * 100 # 100 here to make ftemp = -1 when at 1 cm
            temp = -reward_container['distance_to_goal']/reward_container['start_dist'] # should scale this so that it is -1 at start 
            ftemp,temp = max(ftemp,-2), max(temp, -2)
            # print(reward_container['plane_side'])
            tstep_reward = temp*float(self.config['distance_scaling']) + ftemp*float(self.config['contact_scaling'])
        elif self.config['reward'] == 'SFS':
            tstep_reward = reward_container['slope_to_goal'] * float(self.config['distance_scaling']) - max(reward_container['f1_dist'],reward_container['f2_dist'])*float(self.config['contact_scaling'])
            if (reward_container['distance_to_goal'] < float(self.config['sr'])/1000) & (np.linalg.norm(reward_container['object_velocity']) <= 0.05):
                tstep_reward += float(self.config['success_reward'])
                done2 = True
        elif self.config['reward'] == 'DFS':
            ftemp = max(reward_container['f1_dist'],reward_container['f2_dist'])
            assert ftemp >= 0
            tstep_reward = -reward_container['distance_to_goal'] * float(self.config['distance_scaling'])  - ftemp*float(self.config['contact_scaling'])
            if (reward_container['distance_to_goal'] < float(self.config['sr'])/1000) & (np.linalg.norm(reward_container['object_velocity']) <= 0.05):
                tstep_reward += float(self.config['success_reward'])
                done2 = True
        elif self.config['reward'] == 'SmartDistance + SmartFinger':
            ftemp = max(reward_container['f1_dist'],reward_container['f2_dist'])
            if ftemp > 0.001:
                ftemp = ftemp*ftemp*1000
            temp = -reward_container['distance_to_goal'] * (1 + 4*reward_container['plane_side'])
            tstep_reward = max(temp*float(self.config['distance_scaling']) - ftemp*float(self.config['contact_scaling']),-1)
        else:
            raise Exception('reward type does not match list of known reward types')
        return float(tstep_reward), done2
    
    def draw_multifigure_rewards(self,data_dict):

        episode_number = data_dict['number']

        data = data_dict['timestep_list']
        reward_containers = [f['reward'] for f in data]
        
        overall_reward = []
        f1_reward = []
        f2_reward = []
        dist_reward = []
        for reward in reward_containers:
            overall_reward.append(self.build_reward(reward)[0])
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
            
    def draw_end_poses(self, folder_path):
        print('buckle up') 
        
        # fig, (ax1,ax2) = plt.subplots(2,1,height_ratios=[2,1])

        fig = plt.figure(constrained_layout=True, figsize=(8,6))
        ax = fig.add_gridspec(4, 3)
        ax1 = fig.add_subplot(ax[0:3, :])
        ax2 = fig.add_subplot(ax[-1, :])

        # print('need to load in episode all first')
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
        goals, end_dists, end_poses = [],[], []
        for episode_file in episode_files:
            with open(episode_file, 'rb') as ef:
                tempdata = pkl.load(ef)

            data = tempdata['timestep_list']
            # end_position = data[-1]['state']['obj_2']['pose'][0]
            goals.append(data[0]['state']['goal_pose']['goal_pose'][0:2])
            end_dists.append(data[-1]['reward']['distance_to_goal'])
            end_poses.append(data[-1]['state']['obj_2']['pose'][0])

        bins = np.linspace(0,0.05,100) + 0.05/100
        num_things = np.zeros(100)
        small_thold = max(0.005,min(end_dists))
        med_thold = small_thold+0.005
        big_thold = med_thold + 0.01
        goals = np.array(goals)
        end_poses = np.array(end_poses) - np.array([0,0.1,0])
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

        ax1.scatter(goals[:,0]*100, goals[:,1]*100)
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
            ax1.scatter(med_pose[:,0]*100, med_pose[:,1]*100)
            self.legend.extend(['<= '+str(med_thold*100)+' cm'])
        if len(small_pose)>0:
            small_pose = np.array(small_pose)
            ax1.scatter(small_pose[:,0]*100, small_pose[:,1]*100)
            self.legend.extend(['<= '+str(small_thold*100)+' cm'])
        # ax2.bar(['<0.5 cm','<1 cm','<2 cm','>2 cm'], [len(small_pose),len(med_pose),len(large_pose), len(fucked)])
        ax2.bar(bins, num_things, width=0.05/100)
        plt.tight_layout()
        # self.legend.extend(['Ending Goal Distance'])
        # self.ax.legend(self.legend)
        ax1.set_ylabel('Y position (cm)')
        ax1.set_xlabel('X position (cm)')
        ax1.set_xlim([-7,7])
        ax1.set_ylim([-7,7])
        ax1.set_title('Distance to Goals')
        ax1.grid(False)
        # linea = np.array([[0.0,0.06],[0.0,-0.06]])*100
        # lineb = np.array([[0.0424,-0.0424],[-0.0424,0.0424]])*100
        # linec = np.array([[0.0424,0.0424],[-0.0424,-0.0424]])*100
        # lined = np.array([[0.06,0.0],[-0.06,0.0]])*100
        # ax1.plot(linea[:,0],linea[:,1])
        # ax1.plot(lineb[:,0],lineb[:,1])
        # ax1.plot(linec[:,0],linec[:,1])
        # ax1.plot(lined[:,0],lined[:,1])
        ax1.legend(self.legend)
         
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
        print(finger1_diff,finger2_diff,current_angle_list)
        
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
            goal_poses = len(np.unique([i['state']['goal_pose']['goal_pose'] for i in data], axis=0))
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
            goal_pose = data[-1]['state']['goal_pose']['goal_pose'][0:2]
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
            goal_poses.append(data[-1]['state']['goal_pose']['goal_pose'])
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

    def draw_radar(self,folder_or_data_dict,legend_thing):
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
        goal_poses = []
        name_key_og = [[-0.06,0],[-0.0424,0.0424],[0.0,0.06],[0.0424,0.0424],[0.06,0.0],[0.0424,-0.0424],[0.0,-0.06],[-0.0424,-0.0424]]
        name_key = [[0,0.07],[0.0495,0.0495],[0.07,0.0],[0.0495,-0.0495],[0.0,-0.07],[-0.0495,-0.0495],[-0.07,0.0],[-0.0495,0.0495]]
        name_key2 = ["N","NE","E","SE","S","SW", "W","NW"]
        name_key_og2 = ["E","NE","N","NW","W","SW","S","SE"]
        dist_traveled_list = []
        for episode_file in episode_files:
            with open(episode_file, 'rb') as ef:
                tempdata = pkl.load(ef)

            data = tempdata['timestep_list']
            poses = np.array([i['state']['obj_2']['pose'][0][0:2] for i in data])
            dist_traveled = [poses[i+1]-poses[i] for i in range(len(poses)-1)]
            temp = [np.linalg.norm(d) for d in dist_traveled]
            mag_dist = np.sum(temp)
            dist_traveled_list.append(mag_dist)
            end_poses.append(data[-1]['state']['obj_2']['pose'][0][0:2])
            goal_poses.append(data[-1]['state']['goal_pose']['goal_pose'])
            # if count% 100 ==0:
            #     print('count = ', count)
            count +=1
        end_poses = np.array(end_poses)
        end_poses = end_poses - np.array([0,0.1])
        dist_along_thing = {'E':[],'NE':[],'N':[],'NW':[],'W':[],'SW':[],'S':[],'SE':[]}
        efficiency = {'E':[],'NE':[],'N':[],'NW':[],'W':[],'SW':[],'S':[],'SE':[]}
        for e, g, dt in zip(end_poses, goal_poses, dist_traveled_list):
            for i,name in enumerate(name_key):
                if name == g:
                    dtemp = g/np.linalg.norm(g)*np.dot(e,g/np.linalg.norm(g))
                    dist_along_thing[name_key2[i]].append(dtemp)
                    efficiency[name_key2[i]].append(np.linalg.norm(dtemp)/dt)
            for i,name in enumerate(name_key_og):
                if name == g:
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
        print(f'what we need. mean: {np.sum(alls)/3}, {np.std(alls)}')
        print()
        self.ax.plot(finals[:,0],finals[:,1]+0.1)
        # self.ax.fill(finals[:,0],finals[:,1]+0.1, alpha=0.3)
        self.ax.set_xlim([-0.07,0.07])
        self.ax.set_ylim([0.04,0.16])
        self.ax.set_xlabel('X pos (m)')
        self.ax.set_ylabel('Y pos (m)')
        self.legend.append(legend_thing)
        self.ax.legend(self.legend)
        # self.ax.scatter(end_poses[:,0],end_poses[:,1])