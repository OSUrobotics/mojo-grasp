"""
Created on Tue Dec  6 12:27:24 2022

@author: nigel
"""

import cv2
import imageio
import os
import numpy as np
from plot_scripts import running_plot_and_video
import re
import pickle as pkl

def plot_actor_output_with_video(filepath, episode_number, test=False):
    if test:
        temp = os.listdir(filepath+'Test/')
        filenames = [file for file in temp if str(episode_number)+'.pkl' in temp]
        if len(filenames) > 1:
            raise TypeError('too many files with that number')
        else:
            data_path = filepath+'Test/'+filenames[0]
            file_thing = filenames[0].split('.')[0]
    else:
        data_path = filepath+'Train/episode_'+str(episode_number)+'.pkl'
        file_thing = 'episode_'+str(episode_number)
    video_path = filepath+'Videos/'

    frame_list = []
    for i in range(151):
        frame_path = video_path+file_thing+'_frame_'+str(i)+'.png'
        frame_list.append(frame_path)
    
    x = range(151)
    with open(data_path,'rb') as file:
        data = pkl.load(file)

    actor_list = np.array([f['action']['actor_output'] for f in data['timestep_list']])
    
    running_plot_and_video(frame_list, x, actor_list,
                           xlabel='Elapsed Timesteps', ylabel='Actor Output', title='Actor Output')
    
