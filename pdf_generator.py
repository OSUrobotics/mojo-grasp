#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 10:53:23 2023

@author: orochi
"""
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
# from reportlab.lib import utils
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer
from reportlab.lib.pagesizes import letter
from Data_analysis_gui import GuiBackend
import matplotlib.pyplot as plt
import json
# from PIL import Image
import pathlib
import numpy as np
from io import BytesIO
import os
import cv2
import imageio
class CanvasMock():
    def draw(self):
        pass
    
class ImageMaker(GuiBackend):
    def __init__(self, save_path):
        self.folder = pathlib.Path(save_path)
        self.data_type = None
        self.fig, self.ax = plt.subplots()
        self.figure_canvas_agg = CanvasMock()
        self.figure_canvas_agg.draw()
        self.clear_plots = True
        self.legend = []
        self.curr_graph = None
        self.e_num = -2
        self.all_data = None
        self.moving_avg = 20
        self.colorbar = None
        self.big_data = False
        self.succcess_range = 0.002
        self.use_distance = False
        # self.pdf_canvas = canvas.Canvas(str(self.folder)+'/StandardPlots.pdf', pagesize=letter)
        self.doc = SimpleDocTemplate(str(self.folder)+'/StandardPlots.pdf', pagesize=letter)
        self.images =[]
        self.imnames = []
        self.flowables =[]
        self.count = 0
        

    def plot_to_png(self,fig_name=''):
        self.fig.canvas.draw()
        
        plt.savefig(f'temp{self.count}.png')
        spacer=Spacer(600, 15)
        # buf = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        # buf = buf.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        # imthing = Image.fromarray(buf)
        # self.images.append(imgdata.copy())
        # self.imnames.append(fig_name)
        # letter size is 612x792 points'
        temp = Image(f'temp{self.count}.png')
        # self.pdf_canvas.drawString(100, self.height, fig_name)
        self.flowables.append(Paragraph(fig_name))
        self.flowables.append(temp)
        self.flowables.append(spacer)
        self.count += 1 
        # self.pdf_canvas.drawImage(imgdata, 60, self.height, width=500, height=500)
        # self.pdf_canvas.
        
    def save_pdf(self):
        # for image,name in zip(self.images,self.imnames):
        #     print(type(image))
        #     self.pdf_canvas.drawString(100, 100, name)
        #     self.pdf_canvas.drawImage(image, 30, 600, width=100, height=100)
        # self.pdf_canvas.save()
        self.doc.build(self.flowables)
        files = os.listdir()
        
        files_to_trim = []
        for f in files:
            if 'temp' in f and '.png' in f:
                # print('file to remove')
                print(f)
                os.remove(f)
        
    def plot_sd(self, axes, avg_trial, color, use_filtered=True):

        avg_x, avg_y, _ = avg_trial.get_poses(use_filtered=use_filtered)

        ad_x_up, ad_y_up, _ = avg_trial.get_poses_ad(which_set=1)
        ad_x_down, ad_y_down, _ = avg_trial.get_poses_ad(which_set=2)

        # necessary for building the polygon
        r_ad_x = list(reversed(ad_x_down))
        r_ad_y = list(reversed(ad_y_down))

        poly = []
        for ax, ay in zip(ad_x_up, ad_y_up):
            pt = [ax, ay]
            poly.append(pt)

        # add last point for nicer looking plot
        last_pose = avg_trial.get_last_pose()
        poly.append([last_pose[0], last_pose[1]])

        for ax, ay in zip(r_ad_x, r_ad_y):
            pt = [ax, ay]
            poly.append(pt)

        polyg = plt.Polygon(poly, color=color, alpha=0.4)
        #plt.gca().add_patch(polyg)
        axes.add_patch(polyg)  
        
    def make_gifs(self):
        print("Saving GIF file")
        path = str(self.folder.joinpath('Videos'))+'/'
        filenames = os.listdir(path)
        print(filenames)
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
            with imageio.get_writer(path+key+".gif", mode="I") as writer:
                for idx, frame in enumerate(frames):
                    print("Adding frame to GIF file: ", idx + 1)
                    writer.append_data(frame)
    
    def standard_plots(self):
        train_path = self.folder.joinpath('Train')
        test_path = self.folder.joinpath('Test')
        comparison_path = self.folder.parent.parent.joinpath('resources/GD_runs')
        
        # Training plots
        '''
        print('starting the training plots')
        self.load_data(str(train_path.joinpath('episode_all.pkl')))
        print('loaded episode all')
        self.draw_ending_goal_dist()
        self.plot_to_png('Average Ending Goal Distance')
        self.succcess_range = 0.02
        self.draw_success_rate()
        self.clear_plots = False
        self.succcess_range = 0.01
        self.draw_success_rate()
        self.succcess_range = 0.005
        self.draw_success_rate()
        self.plot_to_png('Average Success Rate')
        self.clear_plots = True
        self.draw_net_reward()
        self.plot_to_png('Average Net Reward')
        '''
        #Asterisk Test in one plot
        #TODO add in the code from cindy stuff
        
        #Asterisk Joint Angle Plots
        print('starting the joint angle plots')
        directions = ['N','NE','E','SE','S','SW','W','NW']
        direction_full_names = ['North','North East','East','South East','South','South West','West', 'North West']
        for direction,nam in zip(directions,direction_full_names):
            self.load_data(str(test_path.joinpath('Asterisk'+direction+'.pkl')))
            self.draw_angles()
            self.clear_plots = False
            self.load_data(str(comparison_path)+'/Asterisk'+nam+'.pkl')
            self.draw_angles()
            self.plot_to_png(nam+' Angles Compared to Gradient Descent')
            self.clear_plots = True
    
        #Actor Output Plots
        print('starting the actor output plots')
        for direction,nam in zip(directions,direction_full_names):
            self.load_data(str(test_path.joinpath('Asterisk'+direction+'.pkl')))
            self.draw_actor_output()
            self.plot_to_png(nam+' Actor Output')
    
        print('saving to pdf')
        self.save_pdf()
        self.make_gifs()

        
    
if __name__ == '__main__':
    a = ImageMaker('/home/orochi/mojo/mojo-grasp/demos/rl_demo/data/PPO_JA_new_physics')
    a.standard_plots()
    # image_path = 'snakehead.jpg'
    # add_image(image_path)