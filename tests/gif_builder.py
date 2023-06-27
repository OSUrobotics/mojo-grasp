"""
Created on Tue Dec  6 12:27:24 2022

@author: nigel
"""

import cv2
import imageio
import os
import numpy as np

# 1. Get Images
print("Saving GIF file")
path = './Videos/'
filenames = os.listdir(path)
direction = [filenam.split('_')[0] for filenam in filenames]
valid_keys = np.unique(direction)
for key in valid_keys:
    frame_names = [file for file in filenames if key+'_' in file]
    tstep = [int(filenam.split('_')[-1].split('.')[0]) for filenam in frame_names]
    tstepinds = np.argsort(tstep)

    frames = []
    
    for ind in tstepinds:
        img = cv2.imread(path+frame_names[ind])
        
        frames.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    with imageio.get_writer(key+".gif", mode="I") as writer:
        for idx, frame in enumerate(frames):
            print("Adding frame to GIF file: ", idx + 1)
            writer.append_data(frame)