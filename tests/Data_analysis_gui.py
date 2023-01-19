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

'''
    Data Plotter
    
    This is based on the Demo_PNG_Viewer by PySimpleGUI
    
'''

def moving_average(a, n) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def save_element_as_file(element, filename):
    """
    Saves any element as an image file.  Element needs to have an underlyiong Widget available (almost if not all of them do)
    :param element: The element to save
    :param filename: The filename to save to. The extension of the filename determines the format (jpg, png, gif, ?)
    """
    widget = element.Widget
    box = (widget.winfo_rootx(), widget.winfo_rooty(), widget.winfo_rootx() + widget.winfo_width(), widget.winfo_rooty() + widget.winfo_height())
    grab = ImageGrab.grab(bbox=box)
    grab.save(filename)

class GuiBackend():
    def __init__(self, canvas):
        self.data_type = None
        self.fig, self.ax = plt.subplots()
        self.figure_canvas_agg = FigureCanvasTkAgg(self.fig, canvas)
        self.figure_canvas_agg.draw()
        self.figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
        self.clear_plots = True
        self.legend = []
        self.curr_graph = None
        self.e_num = -2
        self.all_data = None
        self.moving_avg =1 
        
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
        self.ax.set_ylim([0.1,0.22])
        self.ax.set_xlabel('X pos (m)')
        self.ax.set_ylabel('Y pos (m)')                                                                                                                                                                                                                                   
        self.legend.extend(['RL Trajectory - episode '+str(self.e_num), 'Ideal Path to Goal - episode '+str(self.e_num)])
        self.ax.legend(self.legend)
        self.ax.set_title('Object Path')
        self.figure_canvas_agg.draw()
        self.curr_graph = 'path'

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
        current_action_list= [f['action']['target_joint_angles'] for f in data]
        
        current_angle_list=np.array(current_angle_list)
        current_action_list=np.array(current_action_list)
        angle_tweaks = current_angle_list#current_action_list - current_angle_list
        if self.clear_plots | (self.curr_graph != 'angles'):
            self.ax.cla()
            self.legend = []
        self.ax.plot(range(len(angle_tweaks)),angle_tweaks[:,0])
        self.ax.plot(range(len(angle_tweaks)),angle_tweaks[:,1])
        self.ax.plot(range(len(angle_tweaks)),angle_tweaks[:,2])
        self.ax.plot(range(len(angle_tweaks)),angle_tweaks[:,3])
        self.legend.extend(['Angle 1 - episode '+str(self.e_num), 'Angle 2 - episode '+str(self.e_num), 'Angle 3 - episode '+str(self.e_num), 'Angle 4 - episode '+str(self.e_num)])
        self.ax.legend(self.legend)
        self.ax.set_ylabel('Angle (radians)')
        self.ax.set_xlabel('Timestep (1/240 s)')
        self.ax.set_title('Joint Angles')
        self.figure_canvas_agg.draw()
        self.curr_graph = 'angles'
        
    def draw_actor_output(self):
        if self.e_num == -1:
            print("can't draw when episode_all is selected")
            return
        data = self.data_dict['timestep_list']
        actor_list = [f['control']['actor_output'] for f in data]
        actor_list = np.array(actor_list)
        if self.clear_plots | (self.curr_graph != 'angles'):
            self.ax.cla()
            self.legend = []
        self.ax.plot(range(len(actor_list)),actor_list[:,0])
        self.ax.plot(range(len(actor_list)),actor_list[:,1])
        self.ax.plot(range(len(actor_list)),actor_list[:,2])
        self.ax.plot(range(len(actor_list)),actor_list[:,3])
        self.legend.extend(['Actor Angle 1 - episode ' + str(self.e_num), 'Actor Angle 2 - episode ' + str(self.e_num), 'Actor Angle 3 - episode ' + str(self.e_num), 'Actor Angle 4 - episode ' + str(self.e_num) ])
        self.ax.legend(self.legend)
        self.ax.set_ylabel('Angle (radians)')
        self.ax.set_xlabel('Timestep (1/240 s)')
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
        self.ax.set_ylabel('Action Value')
        self.ax.set_xlabel('Timestep (1/240 s)')
        self.ax.set_title('Critic Output')
        self.figure_canvas_agg.draw()
        self.curr_graph = 'rewards'
    
    def draw_distance_rewards(self):
        if self.e_num == -1:
            print("can't draw when episode_all is selected")
            return
        data = self.data_dict['timestep_list']
        current_reward_dict = [-f['reward']['distance_to_goal'] for f in data]

        if self.clear_plots | (self.curr_graph != 'rewards'):
            self.ax.cla()
            self.legend = []
        self.ax.plot(range(len(current_reward_dict)),current_reward_dict)
        self.legend.extend(['Distance Reward - episode ' + str(self.e_num)])
        self.ax.legend(self.legend)
        self.ax.set_ylabel('Reward')
        self.ax.set_xlabel('Timestep (1/240 s)')
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
        self.legend.extend(['F1 Contact Reward - episode ' + str(self.e_num),'F2 Contact Reward - episode ' + str(self.e_num)])
        self.ax.legend(self.legend)
        self.ax.set_ylabel('Reward')
        self.ax.set_xlabel('Timestep (1/240 s)')
        self.ax.set_title('Contact Reward Plot')
        self.figure_canvas_agg.draw()
        self.curr_graph = 'rewards'
        
    def draw_combined_rewards(self):
        if self.e_num == -1:
            print("can't draw when episode_all is selected")
            return
        data = self.data_dict['timestep_list']
        reward_dict_dist = [-f['reward']['distance_to_goal'] for f in data]
        reward_dict_f1 = [-f['reward']['f1_dist'] for f in data]
        reward_dict_f2 = [-f['reward']['f2_dist'] for f in data]
        reward_dict_penalty = [f['reward']['end_penalty'] for f in data]
        full_reward = []
        for i in range(len(reward_dict_dist)):
            full_reward.append(max(reward_dict_dist[i]+min(reward_dict_f1[i],reward_dict_f2[i])/5,-1))
            
        if self.clear_plots | (self.curr_graph != 'rewards'):
            self.ax.cla()
            self.legend = []
        self.ax.plot(range(len(full_reward)),full_reward)
        self.legend.extend(['Reward - episode ' + str(self.e_num)])
        self.ax.legend(self.legend)
        self.ax.set_ylabel('Reward')
        self.ax.set_xlabel('Timestep (1/240 s)')
        self.ax.set_title('Reward Plot')
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
        nbins=100
        x = datapoints[:,0]
        y = datapoints[:,1]
        print('about to do the gaussian')
        k = kde.gaussian_kde([x,y])
        print('did the gaussian')
        xlim = [np.min(datapoints[:,0]), np.max(datapoints[:,0])]
        ylim = [np.min(datapoints[:,1]), np.max(datapoints[:,1])]
        xlim = [min(xlim[0],-0.07), max(xlim[1],0.07)]
        ylim = [min(ylim[0],0.1), max(ylim[1],0.22)]
        xi, yi = np.mgrid[xlim[0]:xlim[1]:nbins*1j, ylim[0]:ylim[1]:nbins*1j]
        print('did the mgrid')
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        term = -np.max(zi)/5
        for i in range(len(zi)):
            if zi[i] <1:
                zi[i] = term
        self.ax.cla()
        self.legend = []
        print('about to do the colormesh')
        c = self.ax.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto')

        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_xlabel('X pos (m)')
        self.ax.set_ylabel('Y pos (m)')
        self.ax.set_title("explored object poses")
        
        self.fig.colorbar(c, ax=self.ax)
        self.figure_canvas_agg.draw()
        self.curr_graph = 'explored'
        
    def draw_net_reward(self):
        if self.all_data is None:
            print('need to load in episode all first')
            return
        rewards = []
        temp = 0
        for episode in self.all_data['episode_list']:
            data = episode['timestep_list']
            for timestep in data:
                temp += - timestep['reward']['distance_to_goal'] \
                        -max(timestep['reward']['f1_dist'],timestep['reward']['f2_dist'])/5
                                        #timestep['reward']['end_penalty'] \

            rewards.append(temp)
            temp = 0
        if self.moving_avg != 1:
            rewards = moving_average(rewards,self.moving_avg)
        self.ax.cla()
        self.legend = []
        self.ax.plot(range(len(rewards)), rewards)
        self.ax.set_xlabel('Episode Number')
        self.ax.set_ylabel('Total Reward Over the Entire Episode')
        self.ax.set_title("Agent Reward over Episode")
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
            print('need to load in episode all first')
            return
        
        avg_actor_output = np.zeros((len(self.all_data['episode_list']),4))
        avg_actor_std = np.zeros((len(self.all_data['episode_list']),4))
        for i, episode in enumerate(self.all_data['episode_list']):
            data = episode['timestep_list']
            actor_list = [f['control']['actor_output'] for f in data]
            actor_list = np.array(actor_list)
            print(actor_list)
            print(np.average(actor_list, axis=1))
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
        self.legend.extend(['Actor Angle 1 - episode ' + str(self.e_num), 'Actor Angle 2 - episode ' + str(self.e_num), 'Actor Angle 3 - episode ' + str(self.e_num), 'Actor Angle 4 - episode ' + str(self.e_num) ])
        self.ax.legend(self.legend)
        self.ax.set_ylabel('Angle (radians)')
        self.ax.set_xlabel('Timestep (1/240 s)')
        self.ax.set_title('Actor Output')
        self.figure_canvas_agg.draw()
        self.curr_graph = 'angles_total'    
        
    def generate_csv(self):
        if self.e_num != -1:
            print("can't generate data csv unless episode_all is selected")
            return
        # test_dict = {'goal':{'':[]'':[]},'no_goal':{'':[],'':[]},'success_rate':[],'':[],'action_vals':[]}
        training_dict = {}
        for episode in self.data_dict['episode_list']:
            data = episode['timestep_list']
            for timestep in data:
                pass

    def draw_shortest_goal_dist(self):
        if self.all_data is None:
            print('need to load in episode all first')
            return
        closest_dists = np.zeros((len(self.all_data['episode_list']),1))
        for i, episode in enumerate(self.all_data['episode_list']):
            data = episode['timestep_list']
            goal_dist = np.zeros(len(data))
            for j, timestep in enumerate(data):
                goal_dist[j] = timestep['reward']['distance_to_goal']
            closest_dists[i] = np.min(goal_dist, axis=0)

        if self.moving_avg != 1:
            closest_dists = moving_average(closest_dists,self.moving_avg)
        if self.clear_plots | (self.curr_graph != 'goal_dist'):
            self.ax.cla()
            self.legend = []
                
        self.ax.plot(range(len(closest_dists)),closest_dists)
        self.legend.extend(['Min Goal Distance'])
        self.ax.legend(self.legend)
        self.ax.set_ylabel('Goal Distance')
        self.ax.set_xlabel('Episode')
        self.ax.set_title('Distance to Goal Per Episode')
        self.figure_canvas_agg.draw()
        self.curr_graph = 'goal_dist'
        
        
    def draw_ending_velocity(self):
        pass

    def draw_success_rate(self):
        if self.all_data is None:
            print('need to load in episode all first')
            return

        s_f = []
        for i, episode in enumerate(self.all_data['episode_list']):
            data = episode['timestep_list']
            goal_dists = [f['reward']['distance_to_goal'] for f in data]
            ending_dist = min(goal_dists)
            if ending_dist < 0.002:
                s_f.append(100)
            else:
                s_f.append(0)

        if self.moving_avg != 1:
            s_f = moving_average(s_f,self.moving_avg)
        if self.clear_plots | (self.curr_graph != 's_f'):
            self.ax.cla()
            self.legend = []
            
        self.ax.cla()
        self.legend = []        
        self.ax.plot(range(len(s_f)),s_f)
        self.legend.extend(['Success Rate'])
        self.ax.legend(self.legend)
        self.ax.set_ylabel('Success Percentage')
        self.ax.set_xlabel('Episode')
        titlething = 'Percent of Trials over ' + str(self.moving_avg)+' window that are successful'
        self.ax.set_title(titlething)
        self.figure_canvas_agg.draw()
        self.curr_graph = 's_f'

    def draw_path_and_action(self):
        pass

    def draw_ending_goal_dist(self):
        if self.all_data is None:
            print('need to load in episode all first')
            return
        ending_dists = np.zeros((len(self.all_data['episode_list']),1))
        for i, episode in enumerate(self.all_data['episode_list']):
            data = episode['timestep_list']
            ending_dists[i] = np.max(data[-1]['reward']['distance_to_goal'], axis=0)

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
        self.figure_canvas_agg.draw()
        self.curr_graph = 'goal_dist'
        
        
    def show_finger_viz(self):
        pass

    def load_pkl(self, filename):
        
        with open(filename, 'rb') as pkl_file:
            self.data_dict = pkl.load(pkl_file)
            if 'all' in filename:
                self.e_num = -1
                self.all_data = self.data_dict.copy()
            else:
                self.e_num = self.data_dict['number']
    
    def load_json(self, filename):
        with open(filename) as file:
            self.data_dict = json.load(file)
            if 'all' in filename:
                self.e_num = -1
                self.all_data = self.data_dict.copy()
            else:
                self.e_num = self.data_dict['number']
             
    def load_data(self, filename):
        try:
            self.load_pkl(filename)
        except:
            self.load_json(filename)


def main():
    toggle_btn_off = b'iVBORw0KGgoAAAANSUhEUgAAACgAAAAoCAYAAACM/rhtAAAABmJLR0QA/wD/AP+gvaeTAAAED0lEQVRYCe1WTWwbRRR+M/vnv9hO7BjHpElMKSlpqBp6gRNHxAFVcKM3qgohQSqoqhQ45YAILUUVDRxAor2VAweohMSBG5ciodJUSVqa/iikaePEP4nj2Ovdnd1l3qqJksZGXscVPaylt7Oe/d6bb9/svO8BeD8vA14GvAx4GXiiM0DqsXv3xBcJU5IO+RXpLQvs5yzTijBmhurh3cyLorBGBVokQG9qVe0HgwiXLowdy9aKsY3g8PA5xYiQEUrsk93JTtjd1x3siIZBkSWQudUK4nZO1w3QuOWXV+HuP/fL85klAJuMCUX7zPj4MW1zvC0Ej4yMp/w++K2rM9b70sHBYCjo34x9bPelsgp/XJksZ7KFuwZjr3732YcL64ttEDw6cq5bVuCvgy/sje7rT0sI8PtkSHSEIRIKgCQKOAUGM6G4VoGlwiqoVd2Za9Vl8u87bGJqpqBqZOj86eEHGNch+M7otwHJNq4NDexJD+59RiCEQG8qzslFgN8ibpvZNsBifgXmFvJg459tiOYmOElzYvr2bbmkD509e1ylGEZk1Y+Ssfan18n1p7vgqVh9cuiDxJPxKPT3dfGXcN4Tp3dsg/27hUQs0qMGpRMYjLz38dcxS7Dm3nztlUAb38p0d4JnLozPGrbFfBFm79c8hA3H2AxcXSvDz7/+XtZE1kMN23hjV7LTRnKBh9/cZnAj94mOCOD32gi2EUw4FIRUMm6LGhyiik86nO5NBdGRpxYH14bbjYfJteN/OKR7UiFZVg5T27QHYu0RBxoONV9W8KQ7QVp0iXdE8fANUGZa0QAvfhhXlkQcmjJZbt631oIBnwKmacYoEJvwiuFgWncWnXAtuVBBEAoVVXWCaQZzxmYuut68b631KmoVBEHMUUrJjQLXRAQVSxUcmrKVHfjWWjC3XOT1FW5QrWpc5IJdQhDKVzOigEqS5dKHMVplnNOqrmsXqUSkn+YzWaHE9RW1FeXL7SKZXBFUrXW6jIV6YTEvMAUu0W/G3kcxPXP5ylQZs4fa6marcWvvZfJu36kuHjlc/nMSuXz+/ejxgqPFpuQ/xVude9eu39Jxu27OLvBGoMjrUN04zrNMbgVmOBZ96iPdPZmYntH5Ls76KuxL9NyoLA/brav7n382emDfHqeooXyhQmARVhSnAwNNMx5bu3V1+habun5nWdXhwJZ2C5mirTesyUR738sv7g88UQ0rEkTDlp+1wwe8Pf0klegUenYlgyg7bby75jUTITs2rhCAXXQ2vwxz84vlB0tZ0wL4NEcLX/04OrrltG1s8aOrHhk51SaK0us+n/K2xexBxljcsm1n6x/Fuv1PCWGiKOaoQCY1Vb9gWPov50+fdEqd21ge3suAlwEvA14G/ucM/AuppqNllLGPKwAAAABJRU5ErkJggg=='
    toggle_btn_on = b'iVBORw0KGgoAAAANSUhEUgAAACgAAAAoCAYAAACM/rhtAAAABmJLR0QA/wD/AP+gvaeTAAAD+UlEQVRYCe1XzW8bVRCffbvrtbP+2NhOD7GzLm1VoZaPhvwDnKBUKlVyqAQ3/gAkDlWgPeVQEUCtEOIP4AaHSI0CqBWCQyXOdQuRaEFOk3g3IMWO46+tvZ+PeZs6apq4ipON1MNafrvreTPzfvub92bGAOEnZCBkIGQgZOClZoDrh25y5pdjruleEiX+A+rCaQo05bpuvJ/+IHJCSJtwpAHA/e269g8W5RbuzF6o7OVjF8D3Pr4tSSkyjcqfptPDMDKSleW4DKIggIAD5Yf+Oo4DNg6jbUBlvWLUNutAwZu1GnDjzrcXzGcX2AHw/emFUV6Sfk0pqcKpEydkKSo9q3tkz91uF5aWlo1Gs/mYc+i7tz4//19vsW2AU9O381TiioVCQcnlRsWeQhD3bJyH1/MiFLICyBHiuzQsD1arDvypW7DR9nzZmq47q2W95prm+I9fXfqXCX2AF2d+GhI98Y8xVX0lnxvl2UQQg0csb78ag3NjEeD8lXZ7pRTgftmCu4864OGzrq+5ZU0rCa3m+NzXlzvoAoB3+M+SyWQuaHBTEzKMq/3BMbgM+FuFCDBd9kK5XI5PJBKqLSev+POTV29lKB8rT0yMD0WjUSYLZLxzNgZvIHODOHuATP72Vwc6nQ4Uiw8MUeBU4nHS5HA6TYMEl02wPRcZBJuv+ya+UCZOIBaLwfCwQi1Mc4QXhA+PjWRkXyOgC1uIhW5Qd8yG2TK7kSweLcRGKKVnMNExWWBDTQsH9qVmtmzjiThQDs4Qz/OUSGTwcLwIQTLW58i+yOjpXDLqn1tgmDzXzRCk9eDenjo9yhvBmlizrB3V5dDrNTuY0A7opdndStqmaQLPC1WCGfShYRgHdLe32UrV3ntiH9LliuNrsToNlD4kruN8v75eafnSgC6Luo2+B3fGKskilj5muV6pNhk2Qqg5v7lZ51nBZhNBjGrbxfI1+La5t2JCzfD8RF1HTBGJXyDzs1MblONulEqPDVYXgwDIfNx91IUVbAbY837GMur+/k/XZ75UWmJ77ou5mfM1/0x7vP1ls9XQdF2z9uNsPzosXPNFA5m0/EX72TBSiqsWzN8z/GZB08pWq9VeEZ+0bjKb7RTD2i1P4u6r+bwypo5tZUumEcDAmuC3W8ezIqSGfE6g/sTd1W5p5bKjaWubrmWd29Fu9TD0GlYlmTx+8tTJoZeqYe2BZC1/JEU+wQR5TVEUPptJy3Fs+Vkzgf8lemqHumP1AnYoMZSwsVEz6o26i/G9Lgitb+ZmLu/YZtshfn5FZDPBCcJFQRQ+8ih9DctOFvdLIKHH6uUQnq9yhFu0bec7znZ+xpAGmuqef5/wd8hAyEDIQMjAETHwP7nQl2WnYk4yAAAAAElFTkSuQmCC'

    # Get the folder containing the episodes
    folder = sg.popup_get_folder('Episode Folder to open')
    if folder is None:
        sg.popup_cancel('Cancelling')
        return

    # get list of pkl files in folder
    episode_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith('.pkl')]
    filenames_only = [f for f in os.listdir(folder) if f.lower().endswith('.pkl')]
    
    filenums = [re.findall('\d+',f) for f in filenames_only]
    final_filenums = []
    for i in filenums:
        if len(i) > 0 :
            final_filenums.append(int(i[0]))
        else:
            final_filenums.append(10000000000)
    if len(episode_files) == 0:
        sg.popup('No pkl episodes in folder, using json format.')
        episode_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith('.json')]
        filenames_only = [f for f in os.listdir(folder) if f.lower().endswith('.json')]

    sorted_inds = np.argsort(final_filenums)
    final_filenums = np.array(final_filenums)
    temp = final_filenums[sorted_inds]
    episode_files = np.array(episode_files)
    filenames_only = np.array(filenames_only)

    episode_files = episode_files[sorted_inds].tolist()
    filenames_only = filenames_only[sorted_inds].tolist()
    
    
    
    # define menu layout
    menu = [['File', ['Open Folder', 'Exit']], ['Help', ['About', ]]]


    plot_buttons = [[sg.Button('Object Path', size=(8, 2)), sg.Button('Finger Angles', size=(8, 2))],
                    [sg.Button('Actor Output', size=(8, 2)), sg.Button('Critic Output', size=(8, 2)), sg.Button('Rewards', size=(8, 2), key='FullRewards'), sg.Button('Contact Rewards', key='ContactRewards',size=(8, 2)), sg.Button('Distance Rewards', key='SimpleRewards',size=(8, 2))],
                    [sg.Button('Explored Region', size=(8,2)), sg.Button('Episode Rewards', size=(8,2)), sg.Button('Finger Object Avg', size=(8,2)), sg.Button('Finger Object Max', size=(8,2)), sg.Button('Timestep', size=(8,2)), sg.Button('Ending Goal Dist', size=(8,2))],
                    [sg.Button('Shortest Goal Dist', size=(8,2)), sg.Button('Ending Velocity', size=(8,2)), sg.Button('Average Actor Values', size=(8,2)), sg.Button('Success Rate', size=(8,2)), sg.Button('Path + Action', size=(8,2))],
                    [sg.Slider((1,20),10,1,1,key='moving_avg',orientation='h', size=(48,6)), sg.Text("Keep previous graph", size=(10, 3), key='-toggletext-'), sg.Button(image_data=toggle_btn_off, key='-TOGGLE-GRAPHIC-', button_color=(sg.theme_background_color(), sg.theme_background_color()), border_width=0, metadata=False)]]
    # define layout, show and read the window
    col = [[sg.Text(episode_files[0], size=(80, 3), key='-FILENAME-')],
           [sg.Canvas(size=(1280, 960), key='-CANVAS-')],
           plot_buttons[0], plot_buttons[1], plot_buttons[2], plot_buttons[3], plot_buttons[4], [sg.B('Save Image', key='-SAVE-')],
               [sg.Text('File 1 of {}'.format(len(episode_files)), size=(15, 1), key='-FILENUM-')]]

    col_files = [[sg.Listbox(values=filenames_only, size=(60, 30), key='-LISTBOX-', enable_events=True)],
                 [sg.Text('Select an episode.  Use scrollwheel or arrow keys on keyboard to scroll through files one by one.')]]

    layout = [[sg.Menu(menu)], [sg.Col(col_files), sg.Col(col)]]

    window = sg.Window('Analysis Window', layout, return_keyboard_events=True, use_default_focus=False, finalize=True)
    
    canvas = window['-CANVAS-'].TKCanvas

    backend = GuiBackend(canvas)
    # loop reading the user input and displaying image, filename
    filenum, filename = 0, episode_files[0]
    backend.load_data(filename)
    while True:

        event, values = window.read()
        backend.moving_avg = int(values['moving_avg'])
        # --------------------- Button & Keyboard ---------------------
        if event == sg.WIN_CLOSED:
            break
        elif event in ('MouseWheel:Down', 'Down:40', 'Next:34') and filenum < len(episode_files)-1:
            filenum += 1
            filename = os.path.join(folder, filenames_only[filenum])
            window['-LISTBOX-'].update(set_to_index=filenum, scroll_to_index=filenum)
            backend.load_data(filename)
        elif event in ('MouseWheel:Up', 'Up:38', 'Prior:33') and filenum > 0:
            filenum -= 1
            filename = os.path.join(folder, filenames_only[filenum])
            window['-LISTBOX-'].update(set_to_index=filenum, scroll_to_index=filenum)
            backend.load_data(filename)
        elif event == 'Exit':
            break
        elif event == '-LISTBOX-':
            filename = os.path.join(folder, values['-LISTBOX-'][0])
            filenum = episode_files.index(filename)
            backend.load_data(filename)
        elif event == 'Object Path':
            backend.draw_path()
        elif event == 'Finger Angles':
            backend.draw_angles()
        elif event == 'Actor Output':
            backend.draw_actor_output()
        elif event == 'Critic Output':
            backend.draw_critic_output()
        elif event == 'SimpleRewards':
            backend.draw_distance_rewards()
        elif event == 'ContactRewards':
            backend.draw_contact_rewards()
        elif event == 'FullRewards':
            backend.draw_combined_rewards()
        elif event == 'Explored Region':
            backend.draw_explored_region()
        elif event == 'Episode Rewards':
            backend.draw_net_reward()
        elif event == 'Finger Object Avg':
            backend.draw_finger_obj_dist_avg()
        elif event == 'Path + Action':
            backend.draw_path_and_action()
        elif event == 'Success Rate':
            backend.draw_success_rate()
        elif event == 'Average Actor Values':
            backend.draw_avg_actor_output()
        elif event == 'Ending Velocity':
            backend.draw_ending_velocity()
        elif event == 'Shortest Goal Dist':
            backend.draw_shortest_goal_dist()
        elif event == 'Finger Object Max':
            backend.draw_finger_obj_dist_max()
        elif event == 'Timestep':
            backend.draw_timestep_bar_plot()
        elif event == 'Ending Goal Dist':
            backend.draw_ending_goal_dist()   
            
        elif event == '-TOGGLE-GRAPHIC-':  # if the graphical button that changes images
            window['-TOGGLE-GRAPHIC-'].metadata = not window['-TOGGLE-GRAPHIC-'].metadata
            window['-TOGGLE-GRAPHIC-'].update(image_data=toggle_btn_on if window['-TOGGLE-GRAPHIC-'].metadata else toggle_btn_off)
            backend.clear_plots = not backend.clear_plots
        elif event == '-SAVE-':
            filename=r'test.png'
            save_element_as_file(window['-CANVAS-'], filename)
        # ----------------- Menu choices -----------------
        if event == 'Open Folder':
            newfolder = sg.popup_get_folder('New folder', no_window=True)
            if newfolder is None:
                continue

            folder = newfolder
            episode_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith('.pkl')]
            filenames_only = [f for f in os.listdir(folder) if f.lower().endswith('.pkl')]

            window['-LISTBOX-'].update(values=filenames_only)
            window.refresh()

            filenum = 0
        elif event == 'About':
            sg.popup('Demo pkl Viewer Program',
                     'Please give PySimpleGUI a try!')

        # update window with new image
#        window['-IMAGE-'].update(filename=filename)
        # update window with filename
        window['-FILENAME-'].update(filename)
        # update page display
        window['-FILENUM-'].update('File {} of {}'.format(filenum + 1, len(episode_files)))

    window.close()

if __name__ == '__main__':
    main()