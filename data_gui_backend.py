#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 10:04:51 2023

@author: orochi
"""

import os
import pickle as pkl
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import json
from scipy.stats import kde
import re
import time
from matplotlib.patches import Rectangle
from scipy.spatial.transform import Rotation as R
#TODO update this so that the training and evaluation data are saved in separate folders and this gui can deal with both at the same time
def moving_average(a, n) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

class GuiBackend():
    def __init__(self, canvas):
        
        self.folder = None
        self.data_type = None
        self.canvas = canvas
        self.fig, self.ax = plt.subplots()
        # input('stopping before thing 1')
        self.figure_canvas_agg = FigureCanvasTkAgg(self.fig, canvas)
        # input('stopping before thing 2')
        self.figure_canvas_agg.draw()
        # input('stopping before thing 3')
        self.figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
        # input('stopping before thing 4')
        self.clear_plots = True
        self.legend = []
        self.curr_graph = None
        self.e_num = -2
        self.all_data = None
        self.moving_avg = 1 
        self.colorbar = None
        self.big_data = False
        self.succcess_range = 0.002
        self.use_distance = False
        self.min_dists = []
        self.end_dists = []
        self.rewards = []
        self.finger_rewards = []
        self.reduced_format = False
        self.finger = []
        self.distance = []
        self.fig.canvas.callbacks.connect('button_press_event', self.canvas_click)
        self.scatter_tab = False
        self.x_scatter = []
        self.y_scatter = []
        self.config = {}
        
    def draw_path(self):
        if self.e_num == -1:
            print("can't draw when episode_all is selected")
            return
        data = self.data_dict['timestep_list']
        trajectory_points = [f['state']['obj_2']['pose'][0] for f in data]
        goal_pose = data[1]['reward']['goal_position']
        trajectory_points = np.array(trajectory_points)
        if self.clear_plots | (self.curr_graph != 'path'):
            self.ax.cla()
            self.legend = []
        self.ax.plot(trajectory_points[:,0], trajectory_points[:,1])
        self.ax.plot([trajectory_points[0,0], goal_pose[0]],[trajectory_points[0,1],goal_pose[1]])
        self.ax.set_xlim([-0.07,0.07])
        self.ax.set_ylim([0.04,0.16])
        self.ax.set_xlabel('X pos (m)')
        self.ax.set_ylabel('Y pos (m)')                                                                                                                                                                                                                                   
        self.legend.extend(['RL Object Trajectory - episode '+str(self.e_num), 'Ideal Path to Goal - episode '+str(self.e_num)])
        self.ax.legend(self.legend)
        self.ax.set_title('Object Path')
        self.figure_canvas_agg.draw()
        self.curr_graph = 'path'

    def draw_asterisk(self):
        
        # get list of pkl files in folder
        if self.all_data is None:
            # print('need to load in episode all first')
            print('this will be slow, and we both know it')
            episode_files = [os.path.join(self.folder, f) for f in os.listdir(self.folder) if f.lower().endswith('.pkl')]
            filenames_only = [f for f in os.listdir(self.folder) if f.lower().endswith('.pkl')]
            
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
        else:
            goals = []
            trajectories = []
            for episode in self.all_data['episode_list'][-8:]:
                data = episode['timestep_list']
                trajectory_points = [f['state']['obj_2']['pose'][0] for f in data]
                goal = data[1]['reward']['goal_position']
                goals.append(str(np.round(goal[0:2],2)))
                trajectories.append(np.array(trajectory_points))
        if self.clear_plots | (self.curr_graph != 'path'):
            self.ax.cla()
            self.legend = []
        
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
        self.figure_canvas_agg.draw()
        self.curr_graph = 'path'

    def draw_error(self):
        
        data = self.data_dict['timestep_list']
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
        
    def draw_angles(self):
        if self.e_num == -1:
            print("can't draw when episode_all is selected")
            return
        data = self.data_dict['timestep_list']
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
            self.ax.cla()
            self.legend = []
        self.ax.plot(range(len(angle_tweaks)),angle_tweaks[:,0])
        self.ax.plot(range(len(angle_tweaks)),angle_tweaks[:,1])
        self.ax.plot(range(len(angle_tweaks)),angle_tweaks[:,2])
        self.ax.plot(range(len(angle_tweaks)),angle_tweaks[:,3])
        self.legend.extend(['Right Proximal - episode '+str(self.e_num), 
                            'Right Distal - episode '+str(self.e_num), 
                            'Left Proximal - episode '+str(self.e_num), 
                            'Left Distal - episode '+str(self.e_num)])
        self.ax.legend(self.legend)
        self.ax.grid(True)
        self.ax.set_ylabel('Angle (radians)')
        self.ax.set_xlabel('Timestep (1/30 s)')
        self.ax.set_title('Joint Angles')
        self.figure_canvas_agg.draw()
        self.curr_graph = 'angles'
        
    def draw_actor_output(self):
        if self.e_num == -1:
            print("can't draw when episode_all is selected")
            return
        data = self.data_dict['timestep_list']
        actor_list = [f['action']['actor_output'] for f in data]
        actor_list = np.array(actor_list)
        if self.clear_plots | (self.curr_graph != 'angles'):
            self.ax.cla()
            self.legend = []
        self.ax.plot(range(len(actor_list)),actor_list[:,0])
        self.ax.plot(range(len(actor_list)),actor_list[:,1])
        self.ax.plot(range(len(actor_list)),actor_list[:,2])
        self.ax.plot(range(len(actor_list)),actor_list[:,3])
        if self.config['action'] == 'Finger Tip Position':
            self.legend.extend(['Right X - episode ' + str(self.e_num), 
                                'Right Y - episode ' + str(self.e_num), 
                                'Left X - episode ' + str(self.e_num), 
                                'Left Y - episode ' + str(self.e_num)])
        elif self.config['action'] == 'Joint Velocity':            
            self.legend.extend(['Right Proximal - episode '+str(self.e_num), 
                                'Right Distal - episode '+str(self.e_num), 
                                'Left Proximal - episode '+str(self.e_num), 
                                'Left Distal - episode '+str(self.e_num)])
        self.ax.legend(self.legend)
        self.ax.grid(True)
        self.ax.set_ylabel('Actor Output')
        self.ax.set_xlabel('Timestep (1/30 s)')
        self.ax.set_title('Actor Output')
        self.figure_canvas_agg.draw()
        self.curr_graph = 'angles'

    def draw_critic_output(self):
        if self.e_num == -1:
            print("can't draw when episode_all is selected")
            return
        data = self.data_dict['timestep_list']
        critic_list = [f['control']['critic_output'] for f in data]
        print(critic_list[0])
        if self.clear_plots | (self.curr_graph != 'rewards'):
            self.ax.cla()
            self.legend = []
        self.ax.plot(range(len(critic_list)),critic_list)
        self.legend.extend(['Critic Output - episode ' + str(self.e_num)])
        self.ax.legend(self.legend)
        self.ax.grid(True)
        self.ax.set_ylabel('Action Value')
        self.ax.set_xlabel('Timestep (1/30 s)')
        self.ax.set_title('Critic Output')
        self.figure_canvas_agg.draw()
        self.curr_graph = 'rewards'

    def draw_distance_rewards(self):
        if self.e_num == -1:
            print("can't draw when episode_all is selected")
            return
        data = self.data_dict['timestep_list']
        if self.use_distance:
            current_reward_dict = [-f['reward']['distance_to_goal'] for f in data]
        else:
            current_reward_dict = [f['reward']['slope_to_goal'] for f in data]

        if self.clear_plots | (self.curr_graph != 'rewards'):
            self.ax.cla()
            self.legend = []
        self.ax.plot(range(len(current_reward_dict)),current_reward_dict)
        self.legend.extend(['Distance Reward - episode ' + str(self.e_num)])
        self.ax.legend(self.legend)
        self.ax.grid(True)
        self.ax.set_ylabel('Reward')
        self.ax.set_xlabel('Timestep (1/30 s)')
        self.ax.set_title('Reward Plot')
        self.figure_canvas_agg.draw()
        self.curr_graph = 'rewards'
    
    def draw_contact_rewards(self):
        if self.e_num == -1:
            print("can't draw when episode_all is selected")
            return
        data = self.data_dict['timestep_list']
        current_reward_dict1 = [-f['reward']['f1_dist'] for f in data]
        current_reward_dict2 = [-f['reward']['f2_dist'] for f in data]
        if self.clear_plots | (self.curr_graph != 'rewards'):
            self.ax.cla()
            self.legend = []
        self.ax.plot(range(len(current_reward_dict1)),current_reward_dict1)
        self.ax.plot(range(len(current_reward_dict2)),current_reward_dict2)
        self.legend.extend(['Right Finger Contact Reward - episode ' + str(self.e_num),'Left Finger Contact Reward - episode ' + str(self.e_num)])
        self.ax.legend(self.legend)
        self.ax.grid(True)
        self.ax.set_ylabel('Reward')
        self.ax.set_xlabel('Timestep (1/30 s)')
        self.ax.set_title('Contact Reward Plot')
        self.figure_canvas_agg.draw()
        self.curr_graph = 'rewards'
        
    def draw_combined_rewards(self):
        if self.e_num == -1:
            print("can't draw when episode_all is selected")
            return
        data = self.data_dict['timestep_list']
        reward_dict_dist = [f['reward']['distance_to_goal'] for f in data]
        reward_dict_f1 = [f['reward']['f1_dist'] for f in data]
        reward_dict_f2 = [f['reward']['f2_dist'] for f in data]
        current_reward_dict = [f['reward']['slope_to_goal'] for f in data]
        plane_side = [f['reward']['plane_side'] for f in data]
        start_goal = [f['reward']['start_dist'] for f in data]
        full_reward = []
        
        
        for i in range(len(reward_dict_dist)):
            if self.finger[0]:
                ftemp = max(reward_dict_f1[i],reward_dict_f2[i])
            elif self.finger[1]:
                ftemp = max(reward_dict_f1[i],reward_dict_f2[i])
                if ftemp > 0.001:
                    ftemp = ftemp*ftemp*1000
            if self.distance[0]:
                dtemp = reward_dict_dist[i]
            elif self.distance[1]:
                dtemp = reward_dict_dist[i]/start_goal[i]* (1 + 4*plane_side[i])

            elif self.distance[2]:
                dtemp = reward_dict_dist[i] * (1 + 4*plane_side[i])

            elif self.distance[3]:
                dtemp = -current_reward_dict[i]*100
            
            print(f'distance portion: {-dtemp}, finger portion:{-ftemp}')
            full_reward.append(-self.dscale*dtemp - self.fscale*ftemp)
        net_reward = sum(full_reward)
            
        if self.clear_plots | (self.curr_graph != 'rewards'):
            self.ax.cla()
            self.legend = []
        
        title = 'Net Reward: ' + str(net_reward)
        self.ax.plot(range(len(full_reward)),full_reward)
        self.legend.extend(['Reward - episode ' + str(self.e_num)])
        self.ax.legend(self.legend)
        self.ax.set_ylabel('Reward')
        self.ax.set_xlabel('Timestep (1/30 s)')
        self.ax.set_title(title)
        self.ax.grid(True)
        self.figure_canvas_agg.draw()
        self.curr_graph = 'rewards'
         
    def draw_explored_region(self):
        if self.all_data is None:
            print('need to load in episode all first')
            return
        datapoints = []
        for episode in self.all_data['episode_list']:
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
        self.ax.cla()
        self.legend = []
        if self.colorbar:
            self.colorbar.remove()
        print('about to do the colormesh')
        c = self.ax.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto')

        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_xlabel('X pos (m)')
        self.ax.set_ylabel('Y pos (m)')
        self.ax.set_title("Explored Object Poses")
        
        self.colorbar = self.fig.colorbar(c, ax=self.ax)
        self.figure_canvas_agg.draw()
        self.curr_graph = 'explored'

    def draw_end_region(self):
        if self.all_data is None:
            print('need to load in episode all first')
            return
        datapoints = []
        for episode in self.all_data['episode_list']:
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
        self.ax.cla()
        self.legend = []
        if self.colorbar:
            self.colorbar.remove()
        print('about to do the colormesh')
        c = self.ax.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto')

        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_xlabel('X pos (m)')
        self.ax.set_ylabel('Y pos (m)')
        self.ax.set_title("Sampled Object Poses")
        # self.ax.grid(True)
        self.colorbar = self.fig.colorbar(c, ax=self.ax)
        self.figure_canvas_agg.draw()
        self.curr_graph = 'ending_explored'
        
    def draw_sampled_region(self):
        with open(self.folder+'/sampled_positions.pkl', 'rb') as pkl_file:
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
        self.ax.cla()
        self.legend = []
        if self.colorbar:
            self.colorbar.remove()
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
        self.figure_canvas_agg.draw()
        self.curr_graph = 'explored'
    
    def draw_net_reward(self):
        if self.all_data is None:
            if len(self.rewards) ==0:
                # print('need to load in episode all first')
                print('this will be slow, and we both know it')
                
                # get list of pkl files in folder
                episode_files = [os.path.join(self.folder, f) for f in os.listdir(self.folder) if f.lower().endswith('.pkl')]
                filenames_only = [f for f in os.listdir(self.folder) if f.lower().endswith('.pkl')]
                
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
                    individual_rewards = []
                    for tstep in data:
                        individual_rewards.append(self.build_reward(tstep['reward'])[0])
                    rewards.append(sum(individual_rewards))
                    if count% 100 ==0:
                        print('count = ', count)
                    count +=1
                self.rewards= rewards
            else:
                rewards = self.rewards
        else: 
            if self.reduced_format:
                rewards = [-i['sum_dist']-i['sum_finger'] for i in self.all_data['episode_list']]
                print('mew format')
            else:
                rewards = []
                temp = 0
                for episode in self.all_data['episode_list']:
                    data = episode['timestep_list']
                    for timestep in data:
                        if self.use_distance:
                            temp += - timestep['reward']['distance_to_goal'] \
                                    -max(timestep['reward']['f1_dist'],timestep['reward']['f2_dist'])/5
                                                #timestep['reward']['end_penalty'] \
                        else:
                            temp += max(timestep['reward']['slope_to_goal']*100-max(timestep['reward']['f1_dist'],timestep['reward']['f2_dist'])/5,-1)
                    rewards.append(temp)
                    temp = 0
                self.rewards= rewards
        if self.moving_avg != 1:
            rewards = moving_average(rewards,self.moving_avg)
        if self.clear_plots | (self.curr_graph !='Group_Reward'):
            self.ax.cla()
            self.legend = []
        self.legend.append('Average Agent Reward')
        self.ax.plot(range(len(rewards)), rewards)
        self.ax.set_xlabel('Episode Number')
        self.ax.set_ylabel('Total Reward Over the Entire Episode')
        self.ax.set_title("Agent Reward over Episode")
        # self.ax.set_ylim([-12, 0])
        self.ax.legend(self.legend)
        self.ax.grid(True)
        self.figure_canvas_agg.draw()
        self.curr_graph = 'Group_Reward'
        
    def draw_net_distance_reward(self):
        if self.all_data is None:
            if len(self.rewards) ==0:
                # print('need to load in episode all first')
                print('this will be slow, and we both know it')
                
                # get list of pkl files in folder
                episode_files = [os.path.join(self.folder, f) for f in os.listdir(self.folder) if f.lower().endswith('.pkl')]
                filenames_only = [f for f in os.listdir(self.folder) if f.lower().endswith('.pkl')]
                
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
                self.rewards= rewards
            else:
                rewards = self.rewards
        else: 
            if self.reduced_format:
                rewards = [-i['sum_dist'] for i in self.all_data['episode_list']]
            else:
                rewards = []
                temp = 0
                for episode in self.all_data['episode_list']:
                    data = episode['timestep_list']
                    for timestep in data:
                        temp += - timestep['reward']['distance_to_goal']
                    rewards.append(temp)
                    temp = 0
                self.rewards= rewards
        if self.moving_avg != 1:
            rewards = moving_average(rewards,self.moving_avg)
        if self.clear_plots | (self.curr_graph !='Group_Reward'):
            self.ax.cla()
            self.legend = []
        self.legend.append('Average Distance Reward')
        self.ax.plot(range(len(rewards)), rewards)
        self.ax.set_xlabel('Episode Number')
        self.ax.set_ylabel('Total Reward Over the Entire Episode')
        self.ax.set_title("Agent Reward over Episode")
        # self.ax.set_ylim([-12, 0])
        self.ax.legend(self.legend)
        self.ax.grid(True)
        self.figure_canvas_agg.draw()
        self.curr_graph = 'Group_Reward'
        
    def draw_finger_obj_dist_avg(self):
        if self.all_data is None:
            print('need to load in episode all first')
            return
        finger_obj_avgs = np.zeros((len(self.all_data['episode_list']),2))
        for i, episode in enumerate(self.all_data['episode_list']):
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
            self.ax.cla()
            self.legend = []
                
        self.ax.plot(range(len(finger_obj_avgs)),finger_obj_avgs[:,0])
        self.ax.plot(range(len(finger_obj_avgs)),finger_obj_avgs[:,1])
        self.legend.extend(['Average Finger 1 Object Distance', 'Average Finger 2 Object Distance'])
        self.ax.legend(self.legend)
        self.ax.set_ylabel('Finger Object Distance')
        self.ax.set_xlabel('Episode')
        self.ax.grid(True)
        self.ax.set_title('Finger Object Distance Per Episode')
        self.figure_canvas_agg.draw()
        self.curr_graph = 'fing_obj_dist'

    def draw_finger_obj_dist_max(self):
        if self.all_data is None:
            print('need to load in episode all first')
            return
        finger_obj_maxs = np.zeros((len(self.all_data['episode_list']),2))
        for i, episode in enumerate(self.all_data['episode_list']):
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
            self.ax.cla()
            self.legend = []
                
        self.ax.plot(range(len(finger_obj_maxs)),finger_obj_maxs[:,0])
        self.ax.plot(range(len(finger_obj_maxs)),finger_obj_maxs[:,1])
        self.legend.extend(['Maximum Finger 1 Object Distance', 'Maximum Finger 2 Object Distance'])
        self.ax.legend(self.legend)
        self.ax.grid(True)
        self.ax.set_ylabel('Finger Object Distance')
        self.ax.set_xlabel('Episode')
        self.ax.set_title('Finger Object Distance Per Episode')
        self.figure_canvas_agg.draw()
        self.curr_graph = 'fing_obj_dist'
        
    def draw_timestep_bar_plot(self):
        if self.all_data is None:
            print('need to load in episode all first')
            return
        success_timesteps = []
        fail_timesteps = []
        for i, episode in enumerate(self.all_data['episode_list']):
            data = episode['timestep_list']
            num_tsteps = len(data)
            ending_dist = data[-1]['reward']['distance_to_goal']
            if ending_dist < 0.002:
                success_timesteps.append([i, num_tsteps])
            else:
                fail_timesteps.append([i, num_tsteps])
        
        success_timesteps = np.array(success_timesteps)
        fail_timesteps = np.array(fail_timesteps)

        self.ax.cla()
        self.legend = []        
        self.ax.bar(fail_timesteps[:,0],fail_timesteps[:,1])
        if len(success_timesteps) > 0:
            self.ax.bar(success_timesteps[:,0],success_timesteps[:,1])
        self.legend.extend(['Failed Runs', 'Successful Runs'])
        self.ax.legend(self.legend)
        self.ax.set_ylabel('Number of Timesteps')
        self.ax.set_xlabel('Episode')
        self.ax.set_title('Number of Timesteps per Episode')
        self.figure_canvas_agg.draw()
        
    def draw_avg_actor_output(self):
        if self.all_data is None:
            episode_files = [os.path.join(self.folder, f) for f in os.listdir(self.folder) if (f.lower().endswith('.pkl') & ('all' not in f))]
            filenames_only = [f for f in os.listdir(self.folder) if (f.lower().endswith('.pkl') & ('all' not in f))]
            
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
        else:
            avg_actor_output = np.zeros((len(self.all_data['episode_list']),4))
            avg_actor_std = np.zeros((len(self.all_data['episode_list']),4))
            for i, episode in enumerate(self.all_data['episode_list']):
                data = episode['timestep_list']
                actor_list = [f['control']['actor_output'] for f in data]
                actor_list = np.array(actor_list)
                # print(actor_list)
                # print(np.average(actor_list, axis=1))
                avg_actor_output[i,:] = np.average(actor_list, axis = 0)
                avg_actor_std[i,:] = np.std(actor_list, axis = 0)
            
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
            self.ax.cla()
            self.legend = []
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
        self.figure_canvas_agg.draw()
        self.curr_graph = 'angles_total'    
        
    def generate_csv(self):
        if self.e_num != -1:
            print("can't generate data csv unless episode_all is selected")
            return
        # test_dict = {'goal':{'':[]'':[]},'no_goal':{'':[],'':[]},'success_rate':[],'':[],'action_vals':[]}
        for episode in self.data_dict['episode_list']:
            data = episode['timestep_list']
            for timestep in data:
                pass

    def draw_shortest_goal_dist(self):
        if self.all_data is None:
            if len(self.min_dists) == 0:
                # get list of pkl files in folder
                episode_files = [os.path.join(self.folder, f) for f in os.listdir(self.folder) if f.lower().endswith('.pkl')]
                filenames_only = [f for f in os.listdir(self.folder) if f.lower().endswith('.pkl')]
                
                filenums = [re.findall('\d+',f) for f in filenames_only]
                final_filenums = []
                for i in filenums:
                    if len(i) > 0 :
                        final_filenums.append(int(i[0]))
                
                
                sorted_inds = np.argsort(final_filenums)
                final_filenums = np.array(final_filenums)
                filenames_only = np.array(filenames_only)
    
                episode_files = episode_files[sorted_inds].tolist()
                
                min_dists = []
                for i, episode_file in enumerate(episode_files):
                    with open(episode_file, 'rb') as ef:
                        tempdata = pkl.load(ef)
                    data = tempdata['timestep_list']
                    goal_dists = [f['reward']['distance_to_goal'] for f in data]
                    min_dists.append(min(goal_dists))
                    if i % 100 ==0:
                        print('count = ',i)
                self.min_dists = min_dists
            else:
                min_dists = self.min_dists
        else:
            if self.reduced_format:
                min_dists = [i['min_dist'] for i in self.all_data['episode_list']]
            else:
                min_dists = np.zeros((len(self.all_data['episode_list']),1))
                for i, episode in enumerate(self.all_data['episode_list']):
                    data = episode['timestep_list']
                    goal_dist = np.zeros(len(data))
                    for j, timestep in enumerate(data):
                        goal_dist[j] = timestep['reward']['distance_to_goal']
                    min_dists[i] = np.min(goal_dist, axis=0)

        if self.moving_avg != 1:
            min_dists = moving_average(min_dists,self.moving_avg)
        if self.clear_plots | (self.curr_graph != 'goal_dist'):
            self.ax.cla()
            self.legend = []
                
        self.ax.plot(range(len(min_dists)),min_dists)
        self.legend.extend(['Min Goal Distance'])
        self.ax.legend(self.legend)
        self.ax.set_ylabel('Goal Distance')
        self.ax.set_xlabel('Episode')
        self.ax.grid(True)
        self.ax.set_title('Distance to Goal Per Episode')
        self.figure_canvas_agg.draw()
        self.curr_graph = 'goal_dist'
            
    def draw_ending_velocity(self):
        if self.all_data is None:
            print('need to load in episode all first')
            return

        # thought 1
        '''
        velocity = np.zeros((len(self.all_data['episode_list']),2))
        pose = np.zeros((len(self.all_data['episode_list']),2))
        for i, episode in enumerate(self.all_data['episode_list']):
            data = episode['timestep_list']
            obj_poses = [f['state']['obj_2']['pose'][0] for f in data[-5:]]
            obj_poses = np.array(obj_poses)
            print(obj_poses)
            try:
                slope, _ = np.polyfit(obj_poses[:,0],obj_poses[:,1], 1)
                print('slope',slope)
                dx = obj_poses[-1,0] - obj_poses[0,0]
                dy = dx * slope
            except np.linalg.LinAlgError:
                dx = 0.00001
                dy = 0.00001
            pose[i,:] = obj_poses[-1,0:2]
            velocity[i,:] = [dx,dy]

        if self.clear_plots | (self.curr_graph != 's_f'):
            self.ax.cla()
            self.legend = []
            
        topbounds = np.max(pose, axis = 1)
        botbounds = np.min(pose, axis = 1)
        self.ax.cla()
        self.legend = []        
        self.ax.quiver(pose[:,0],pose[:,1],velocity[:,0],velocity[:,1])
        self.ax.set_ylabel('y')
        self.ax.set_xlabel('x')
        self.ax.set_xlim([min(-0.07,botbounds[0]),max(0.07,topbounds[0])])
        self.ax.set_ylim([min(0.1,botbounds[1]),max(0.22,topbounds[1])])
        self.ax.set_title("Ending Velocity")
        self.figure_canvas_agg.draw()
        self.curr_graph = 'vel'
        '''
        # thought 2

        velocity = np.zeros((len(self.all_data['episode_list']),2))
        for i, episode in enumerate(self.all_data['episode_list']):
            data = episode['timestep_list']
            obj_poses = [f['state']['obj_2']['pose'][0] for f in data[-5:]]
            obj_poses = np.array(obj_poses)
            dx = obj_poses[-1,0] - obj_poses[0,0]
            dy = obj_poses[-1,1] - obj_poses[0,1]
            velocity[i,:] = [dx,dy]

        if self.clear_plots | (self.curr_graph != 's_f'):
            self.ax.cla()
            self.legend = []
        ending_vel = np.sqrt(velocity[:,0]**2 + velocity[:,1]**2)
        if self.moving_avg != 1:
            ending_vel = moving_average(ending_vel,self.moving_avg)
        self.ax.cla()
        self.legend = []        
        self.ax.plot(range(len(ending_vel)),ending_vel)
        self.ax.set_ylabel('ending velocity magnitude')
        self.ax.set_xlabel('episode')
        self.ax.grid(True)
        self.ax.set_title("Ending Velocity")
        self.figure_canvas_agg.draw()
        self.curr_graph = 'vel'

    def draw_success_rate(self):
        if self.all_data is None:
            # print('this will be slow, and we both know it')
            if len(self.min_dists) == 0:
                # get list of pkl files in folder
                episode_files = [os.path.join(self.folder, f) for f in os.listdir(self.folder) if f.lower().endswith('.pkl')]
                filenames_only = [f for f in os.listdir(self.folder) if f.lower().endswith('.pkl')]
                
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
                
                min_dists = []
                
                for i, episode_file in enumerate(episode_files):
                    with open(episode_file, 'rb') as ef:
                        tempdata = pkl.load(ef)
                    data = tempdata['timestep_list']
                    goal_dists = [f['reward']['distance_to_goal'] for f in data]
                    min_dists.append(min(goal_dists))
                    if i % 100 ==0:
                        print('count = ',i)
                self.min_dists = min_dists
            s_f = []
            for dist in self.min_dists:
                if dist < self.succcess_range:
                    s_f.append(100)
                else:
                    s_f.append(0)

        else:
            if self.reduced_format:
                s_f = [100*(i['min_dist']<self.succcess_range) for i in self.all_data['episode_list']]
            else:
                s_f = []
                min_dists = []
                for i, episode in enumerate(self.all_data['episode_list']):
                    data = episode['timestep_list']
                    goal_dists = [f['reward']['distance_to_goal'] for f in data]
                    ending_dist = min(goal_dists)
                    min_dists.append(ending_dist)
                    if ending_dist < self.succcess_range:
                        s_f.append(100)
                    else:
                        s_f.append(0)
                self.min_dists = min_dists
            
        if self.moving_avg != 1:
            s_f = moving_average(s_f,self.moving_avg)
        if self.clear_plots | (self.curr_graph != 's_f'):
            self.ax.cla()
            self.legend = []
        self.ax.plot(range(len(s_f)),s_f)
        self.legend.extend(['Success Rate (' + str(self.succcess_range*1000) + ' mm tolerance)'])
        self.ax.legend(self.legend)
        self.ax.set_ylabel('Success Percentage')
        self.ax.set_xlabel('Episode')
        self.ax.set_ylim([-1,101])
        titlething = 'Percent of Trials over ' + str(self.moving_avg)+' window that are successful'
        self.ax.set_title(titlething)
        self.ax.grid(True)
        self.figure_canvas_agg.draw()
        self.curr_graph = 's_f'
        pass

    def draw_ending_goal_dist(self):
        if self.all_data is None:
            if len(self.end_dists) == 0:
                # print('need to load in episode all first')
                episode_files = [os.path.join(self.folder, f) for f in os.listdir(self.folder) if f.lower().endswith('.pkl')]
                filenames_only = [f for f in os.listdir(self.folder) if f.lower().endswith('.pkl')]
                
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
                self.end_dists = ending_dists
            else:
                ending_dists = self.end_dists
        else:
            if self.reduced_format:
                ending_dists = [i['ending_dist'] for i in self.all_data['episode_list']]
            else:
                ending_dists = np.zeros((len(self.all_data['episode_list']),1))
                for i, episode in enumerate(self.all_data['episode_list']):
                    data = episode['timestep_list']
                    ending_dists[i] = np.max(data[-1]['reward']['distance_to_goal'], axis=0)
                self.end_dists = ending_dists

        if self.moving_avg != 1:
            ending_dists = moving_average(ending_dists,self.moving_avg)
            
        if self.clear_plots | (self.curr_graph != 'goal_dist'):
            self.ax.cla()
            self.legend = []
                
        self.ax.plot(range(len(ending_dists)),ending_dists)
        self.legend.extend(['Ending Goal Distance'])
        self.ax.legend(self.legend)
        self.ax.set_ylabel('Goal Distance')
        self.ax.set_xlabel('Episode')
        self.ax.set_title('Distance to Goal Per Episode')
        self.ax.grid(True)
        self.figure_canvas_agg.draw()
        self.curr_graph = 'goal_dist'
        
    def draw_goal_rewards(self): # Depreciated
        if self.all_data is None:
            print('need to load in episode all first')
            # print('this will be slow, and we both know it')
            return
        
        keylist = ['forward','backward', 'forwardleft', 'backwardleft','forwardright', 'backwardright', 'left', 'right']
        rewards = {'forward':[[0.0, 0.2],[]],'backward':[[0.0, 0.12],[]],'forwardleft':[[-0.03, 0.19],[]],'backwardleft':[[-0.03,0.13],[]],
                   'forwardright':[[0.03,0.19],[]],'backwardright':[[0.03, 0.13],[]],'left':[[-0.04, 0.16],[]],'right':[[0.04,0.16],[]]}
        # sucessful_dirs = []
        for i, episode in enumerate(self.all_data['episode_list']):
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
            self.ax.cla()
            self.legend = []
            
        self.ax.plot(range(len(rewards[keylist[best]][1])),rewards[keylist[best]][1])
        self.ax.plot(range(len(rewards[keylist[worst]][1])),rewards[keylist[worst]][1])
        self.legend.extend(['Best Direction: ' + keylist[best], 'Worst Direction: ' + keylist[worst]])
        self.ax.legend(self.legend)
        self.ax.set_ylabel('Net Reward')
        self.ax.set_xlabel('Episode')
        self.ax.set_title('Net Reward By Direction')
        self.ax.grid(True)
        self.figure_canvas_agg.draw()
        self.curr_graph = 'direction_success_thing' 

    def draw_actor_max_percent(self):
        if len(self.min_dists) == 0:
            # get list of pkl files in folder
            episode_files = [os.path.join(self.folder, f) for f in os.listdir(self.folder) if f.lower().endswith('.pkl')]
            filenames_only = [f for f in os.listdir(self.folder) if f.lower().endswith('.pkl')]
            
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
            self.ax.cla()
            self.legend = []
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
        self.figure_canvas_agg.draw()
        self.curr_graph = 'angles'
    
    def draw_goal_s_f(self):
        if self.all_data is None:
            print('need to load in episode all first')
            # print('this will be slow, and we both know it')
            return
        
        rewards = {'forward':[[0.0, 0.2],[]],'backward':[[0.0, 0.12],[]],'forwardleft':[[-0.03, 0.19],[]],'backwardleft':[[-0.03,0.13],[]],
                   'forwardright':[[0.03,0.19],[]],'backwardright':[[0.03, 0.13],[]],'left':[[-0.04, 0.16],[]],'right':[[0.04,0.16],[]]}
        # sucessful_dirs = []
        for i, episode in enumerate(self.all_data['episode_list']):
            data = episode['timestep_list']
            goal_dists = [f['reward']['distance_to_goal'] for f in data]
            ending_dist = min(goal_dists)
            goal_pose = data[1]['reward']['goal_position'][0:2]
            if ending_dist < self.succcess_range:
                temp = 100
            else:
                temp = 0
            for i, v in rewards.items():
                if np.isclose(goal_pose, v[0]).all():
                    v[1].append(temp)

        
        # s = np.unique(sucessful_dirs)
        # print('succesful directions', s)

        # if self.moving_avg != 1:
        #     closest_dists = moving_average(closest_dists,self.moving_avg)
        sf = []
        reduced_key_list = ['forward','backward','left','right']
        if self.moving_avg != 1:
            for i in reduced_key_list:
                sf.append(moving_average(rewards[i][1],self.moving_avg))
        if self.clear_plots | (self.curr_graph != 's_f'):
            self.ax.cla()
            self.legend = []
        for i,yax in enumerate(sf):
            self.ax.plot(range(len(yax)),yax)
            self.legend.extend([reduced_key_list[i]])
        self.ax.legend(self.legend)
        self.ax.set_ylabel('Net Reward')
        self.ax.set_xlabel('Episode')
        self.ax.set_title('Net Reward By Direction')
        self.ax.grid(True)
        self.figure_canvas_agg.draw()
        self.curr_graph = 'direction_reward_thing'   
    
    def draw_fingertip_path(self):
        if self.e_num == -1:
            print("can't draw when episode_all is selected")
            return
        data = self.data_dict['timestep_list']
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
        print(arrow_points, next_points)
        self.ax.cla()
        self.legend = []
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
        self.ax.set_xlim([-0.07,0.07])
        self.ax.set_ylim([0.04,0.16])
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
        self.ax.set_title('Object and Finger Path - Episode: '+str(self.e_num))
        self.figure_canvas_agg.draw()
        self.curr_graph = 'path'

    def draw_obj_contacts(self):
        if self.e_num == -1:
            print("can't draw when episode_all is selected")
            return
        data = self.data_dict['timestep_list']
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
        self.ax.cla()
        self.legend = []
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
        self.ax.set_title('Object and Finger Path - Episode: '+str(self.e_num))
        self.figure_canvas_agg.draw()
        self.curr_graph = 'path'

    def draw_net_finger_reward(self):
        if self.all_data is None:
            if len(self.finger_rewards) ==0:
                # print('need to load in episode all first')
                print('this will be slow, and we both know it')
                
                # get list of pkl files in folder
                episode_files = [os.path.join(self.folder, f) for f in os.listdir(self.folder) if f.lower().endswith('.pkl')]
                filenames_only = [f for f in os.listdir(self.folder) if f.lower().endswith('.pkl')]
                
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
                        temp += -max(timestep['reward']['f1_dist'],timestep['reward']['f2_dist'])/5
                    rewards.append(temp)
                    temp = 0
                    if count% 100 ==0:
                        print('count = ', count)
                    count +=1
                self.finger_rewards= rewards
            else:
                rewards = self.finger_rewards
        else: 
            if self.reduced_format:
                rewards = [-i['sum_finger'] for i in self.all_data['episode_list']]
                print('new one')
            else:
                rewards = []
                temp = 0
                for episode in self.all_data['episode_list']:
                    data = episode['timestep_list']
                    for timestep in data:
                        temp += -max(timestep['reward']['f1_dist'],timestep['reward']['f2_dist'])/5
                    rewards.append(temp)
                    temp = 0
                self.finger_rewards= rewards
        if self.moving_avg != 1:
            rewards = moving_average(rewards,self.moving_avg)
        if self.clear_plots | (self.curr_graph !='Group_Reward'):
            self.ax.cla()
        self.legend = []
        self.ax.plot(range(len(rewards)), rewards)
        self.ax.set_xlabel('Episode Number')
        self.ax.set_ylabel('Total Reward Over the Entire Episode')
        self.ax.set_title("Agent Reward over Episode")
        self.legend.append('Average Finger Tip Reward')
        self.ax.legend(self.legend)
        # self.ax.set_ylim([-0.61, 0.001])
        self.ax.grid(True)
        self.figure_canvas_agg.draw()
        self.curr_graph = 'Group_Reward'
    
    def check_all_data(self):
        #TODO fix this so that it works with different shit

        if self.all_data['episode_list'][0].keys():
            print('Episode_all is in reduced format, some plotting functions may be unavailable')
            self.reduced_format = True
        else:
            num_episodes = len(self.all_data['episode_list'])
            self.reduced_format = False
            for i in range(num_episodes):
                assert i+1 == self.all_data['episode_list'][i]['number']
            
    def show_finger_viz(self):
        pass

    def load_pkl(self, filename):
        self.folder = os.path.dirname(filename)
        with open(filename, 'rb') as pkl_file:
            self.data_dict = pkl.load(pkl_file)
            if 'all.' in filename:
                self.e_num = -1
                self.all_data = self.data_dict.copy()
                self.check_all_data()
            else:
                self.e_num = self.data_dict['number']
    
    def load_json(self, filename):
        with open(filename) as file:
            self.data_dict = json.load(file)
            if 'all.' in filename:
                self.e_num = -1
                self.all_data = self.data_dict.copy()
            else:
                self.e_num = self.data_dict['number']
             
    def load_data(self, filename):
        # try:
        self.load_pkl(filename)
        
    def load_config(self, config_folder):
        with open(config_folder+'/experiment_config.json') as file:
            self.config = json.load(file)
            
    def draw_scatter_end_dist(self):
        
        self.figure_canvas_agg.draw()
        # print('need to load in episode all first')
        episode_files = [os.path.join(self.folder, f) for f in os.listdir(self.folder) if f.lower().endswith('.pkl')]
        filenames_only = [f for f in os.listdir(self.folder) if f.lower().endswith('.pkl')]
        
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
            end_dists.append(data[-1]['reward']['distance_to_goal'])
        if self.colorbar:
            self.colorbar.remove()        
        self.ax.cla()
        self.legend = []
        
        goals = np.array(goals)
        end_dists = np.array(end_dists)

        a = self.ax.scatter(goals[:,0]*100, goals[:,1]*100, c = end_dists*100, cmap='jet')
        # self.legend.extend(['Ending Goal Distance'])
        # self.ax.legend(self.legend)
        self.ax.set_ylabel('Y position (cm)')
        self.ax.set_xlabel('X position (cm)')
        self.ax.set_xlim([-7,7])
        self.ax.set_ylim([-7,7])
        self.ax.set_title('Distance to Goals')
        self.ax.grid(False)
        self.colorbar = self.fig.colorbar(a, ax=self.ax)
        self.figure_canvas_agg.draw()
        self.curr_graph = 'scatter'
        
    def draw_scatter_contact_dist(self):
        
        # print('need to load in episode all first')
        episode_files = [os.path.join(self.folder, f) for f in os.listdir(self.folder) if f.lower().endswith('.pkl')]
        filenames_only = [f for f in os.listdir(self.folder) if f.lower().endswith('.pkl')]
        
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
        if self.colorbar:
            self.colorbar.remove()
        self.ax.cla()
        self.legend = []
        
        goals = np.array(goal_positions)
        finger_max_dists = np.array(finger_max_dists)
        a = self.ax.scatter(goals[:,0]*100, goals[:,1]*100, c = finger_max_dists*100, cmap='jet')
        # self.legend.extend(['Ending Goal Distance'])
        # self.ax.legend(self.legend)
        self.ax.set_ylabel('Y position (cm)')
        self.ax.set_xlabel('X position (cm)')
        self.ax.set_xlim([-7,7])
        self.ax.set_ylim([-7,7])
        self.ax.set_title('Maximum Finger Distance')
        self.ax.grid(False)
        self.colorbar = self.fig.colorbar(a, ax=self.ax)
        self.figure_canvas_agg.draw()
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
        elif self.config['reward'] == 'SFS':
            tstep_reward = reward_container['slope_to_goal'] * float(self.config['distance_scaling']) - max(reward_container['f1_dist'],reward_container['f2_dist'])*float(self.config['contact_scaling'])
            if (reward_container['distance_to_goal'] < float(self.config['sr'])/1000) & (np.linalg.norm(reward_container['object_velocity']) <= 0.05):
                tstep_reward += float(self.config['success_reward'])
                done2 = True
                print('SUCCESS BABY!!!!!!!')
        elif self.config['reward'] == 'DFS':
            ftemp = max(reward_container['f1_dist'],reward_container['f2_dist'])
            assert ftemp >= 0
            tstep_reward = -reward_container['distance_to_goal'] * float(self.config['distance_scaling'])  - ftemp*float(self.config['contact_scaling'])
            if (reward_container['distance_to_goal'] < float(self.config['sr'])/1000) & (np.linalg.norm(reward_container['object_velocity']) <= 0.05):
                tstep_reward += float(self.config['success_reward'])
                done2 = True

                print('SUCCESS BABY!!!!!!!')
        elif self.config['reward'] == 'SmartDistance + SmartFinger':
            ftemp = max(reward_container['f1_dist'],reward_container['f2_dist'])
            if ftemp > 0.001:
                ftemp = ftemp*ftemp*1000
            temp = -reward_container['distance_to_goal'] * (1 + 4*reward_container['plane_side'])
            tstep_reward = max(temp*float(self.config['distance_scaling']) - ftemp*float(self.config['contact_scaling']),-1)
        else:
            raise Exception('reward type does not match list of known reward types')
        return float(tstep_reward), done2
    
    def draw_multifigure_rewards(self,canvas):
        if self.e_num == -1:
            print("can't draw when episode_all is selected")
            return
        data = self.data_dict['timestep_list']
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
        
        if self.clear_plots | (self.curr_graph != 'rewards'):
            self.ax.cla()
            self.legend = []

        data = self.data_dict['timestep_list']
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
        figure_canvas_agg2 = FigureCanvasTkAgg(fig, canvas)
        figure_canvas_agg2.draw()
        figure_canvas_agg2.get_tk_widget().pack(side='top', fill='both', expand=1)
        
        axes[0,1].plot(trajectory_points[:,0], trajectory_points[:,1])
        axes[0,1].plot(fingertip1_points[:,0], fingertip1_points[:,1])
        axes[0,1].plot(fingertip2_points[:,0], fingertip2_points[:,1])
        axes[0,1].plot([trajectory_points[0,0], goal_pose[0]],[trajectory_points[0,1],goal_pose[1]])
        axes[0,1].set_xlim([-0.07,0.07])
        axes[0,1].set_ylim([0.04,0.16])
        axes[0,1].set_xlabel('X pos (m)')
        axes[0,1].set_ylabel('Y pos (m)')
        
        axes[0,0].plot(range(len(f1_reward)), -f1_reward)
        axes[0,0].plot(range(len(f1_reward)), -f2_reward)
        axes[1,0].plot(range(len(f1_reward)), -dist_reward)
        axes[1,1].plot(range(len(f1_reward)), overall_reward)
        axes[0,0].set_title('Contact Distances')
        axes[1,0].set_title('Object Goal Distance')
        axes[1,1].set_title('Total Reward')
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
        axes[0,1].set_title('Object and Finger Path - Episode: '+str(self.e_num))
        plt.tight_layout()
        self.figure_canvas_agg.draw()
        self.curr_graph = 'path'
        figure_canvas_agg2.draw()
            
    def draw_end_poses(self, canvas):
        print('buckle up') 
        
        fig, (ax1,ax2) = plt.subplots(2,1)
        figure_canvas_agg2 = FigureCanvasTkAgg(fig, canvas)
        figure_canvas_agg2.draw()
        figure_canvas_agg2.get_tk_widget().pack(side='top', fill='both', expand=1)
        
        # print('need to load in episode all first')
        episode_files = [os.path.join(self.folder, f) for f in os.listdir(self.folder) if f.lower().endswith('.pkl')]
        filenames_only = [f for f in os.listdir(self.folder) if f.lower().endswith('.pkl')]
        
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
        
        small_pose = np.array(small_pose)

        ax1.scatter(goals[:,0]*100, goals[:,1]*100)
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
        ax1.scatter(small_pose[:,0]*100, small_pose[:,1]*100)
        self.legend.extend(['<= '+str(small_thold*100)+' cm'])
        ax2.bar(['<0.5 cm','<1 cm','<2 cm','>2 cm'], [len(small_pose),len(med_pose),len(large_pose), len(fucked)])
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
        self.figure_canvas_agg.draw()
        self.curr_graph = 'scatter'
        figure_canvas_agg2.draw()
    
    def canvas_click(self, event):
        print(event.xdata, event.ydata)
        if self.scatter_tab:
            distances = [(x - event.xdata)**2 + (y - event.ydata)**2 for x, y in zip(self.x_scatter, self.y_scatter)]
            closest_point_index = min(range(len(distances)), key=distances.__getitem__)
            print(closest_point_index)