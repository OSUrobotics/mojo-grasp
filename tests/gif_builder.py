"""
Created on Tue Dec  6 12:27:24 2022

@author: nigel
"""

import cv2
# import imageio
import os
import numpy as np
# from plot_scripts import running_plot_and_video
import re
import pickle as pkl
# import glob
from PIL import Image

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
    frames = [Image.open(image) for image in frame_list]
    frame_one = frames[0]
    frame_one.save("Episode.gif", format="GIF", append_images=frames,
               save_all=True, duration=151, loop=0)
    '''
    x = range(151)
    with open(data_path,'rb') as file:
        data = pkl.load(file)

    eival_list = np.array([f['state'][
        'two_finger_gripper']['eigenvalues'] for f in data['timestep_list']])
    eival_ratio1 = np.max(eival_list[0:2],axis=1)/np.min(eival_list[0:2],axis=1)
    eival_ratio2 = np.max(eival_list[2:4],axis=1)/np.min(eival_list[2:4],axis=1)
    running_plot_and_video(frame_list, x, actor_list,
                           xlabel='Elapsed Timesteps', ylabel='Actor Output', title='Actor Output')
    '''

def make_video(filepath):
    temp = os.listdir(filepath+'Videos/')
    video_path = filepath+'Videos/'
    file_thing = 'eval'
    frame_list = []
    frames = []
    for i in range(25):
        for j in range(8):
            print(video_path)
            frame_path = video_path+file_thing+'_frame_'+str(i)+'_'+str(j*10)+'.png'
            frame_list.append(frame_path)
    frames = [Image.open(image) for image in frame_list]
    frame_one = frames[0]
    frame_one.save(filepath+"Episode.gif", format="GIF", append_images=frames,
               save_all=True, duration=13*8, loop=0)

def make_gif_from_video(filepath,filename):
    cap = cv2.VideoCapture(filepath+filename)
 
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    frames = []
    # Read until video is completed
    frame_count = 0
    while(cap.isOpened()):
    # Capture frame-by-frame
        frame_count+=1
        ret, frame = cap.read()
        # print('got to here', ret)
        if ret == True:
            if frame_count %2 == 0:
                frames.append(Image.fromarray(frame))
        else:
            break
    frame_one = frames[0]
    print(len(frames))
    input('make into a gif?')
    frame_one.save(filepath+"Episode.gif", format="GIF", append_images=frames,
               save_all=True, duration=len(frames), loop=0)
        
    # When everything done, release the video capture object
    cap.release()
    
    # Closes all the frames
    cv2.destroyAllWindows()

# make_video('/home/mothra/mojo-grasp/demos/rl_demo/data/Mothra_Rotation/JA_S3/')
make_gif_from_video('/home/mothra/','trimmed_video.mp4')
