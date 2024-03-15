#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 10:53:23 2023

@author: orochi
"""
from reportlab.lib.pagesizes import letter, landscape
from reportlab.pdfgen import canvas
# from reportlab.lib import utils
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer, Table
from mojograsp.simcore.data_gui_backend import PlotBackend
import matplotlib.pyplot as plt
import json
# from PIL import Image
import pathlib
import numpy as np
from io import BytesIO
import os
import cv2
import csv
# import imageio
class CanvasMock():
    def draw(self):
        pass
    
class ImageMaker():
    def __init__(self, save_path,not_stand=False,land=False):
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
        self.colorbar = None
        self.big_data = False
        self.succcess_range = 0.002
        self.use_distance = False
        self.backend = PlotBackend(str(self.folder))
        self.backend.moving_avg = 20
        self.table = []
        self.tablen = 8
        # self.pdf_canvas = canvas.Canvas(str(self.folder)+'/StandardPlots.pdf', pagesize=letter)
        if landscape:
            if not_stand:
                self.doc = SimpleDocTemplate(str(self.folder)+'/OtherPlots.pdf', pagesize=landscape(letter), )
            else:
                self.doc = SimpleDocTemplate(str(self.folder)+'/StandardPlots.pdf', pagesize=landscape(letter))
        else:
            if not_stand:
                self.doc = SimpleDocTemplate(str(self.folder)+'/OtherPlots.pdf', pagesize=letter, )
            else:
                self.doc = SimpleDocTemplate(str(self.folder)+'/StandardPlots.pdf', pagesize=letter)
        self.images =[]
        self.imnames = []
        self.flowables =[]
        self.count = 0
        self.thing = False

    def plot_to_png(self,fig_name=''):
        self.fig.canvas.draw()
        
        plt.savefig(f'temp{self.count}.png')
        
        # spacer=Spacer(600, 15)
        # buf = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        # buf = buf.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        # imthing = Image.fromarray(buf)
        # self.images.append(imgdata.copy())
        # self.imnames.append(fig_name)
        # letter size is 612x792 points'
        
        # self.pdf_canvas.drawString(100, self.height, fig_name)
        if not self.thing:
            self.table.append([Paragraph(fig_name)])
            self.table.append([Image(f'temp{self.count}.png',width=300,height = 200,hAlign='LEFT')])
        if self.thing:
            print(self.count, len(self.table))
            
            self.table[(self.count-self.tablen)*2].append(Paragraph(fig_name))
            self.table[(self.count-self.tablen)*2 + 1].append(Image(f'temp{self.count}.png',width=300,height = 200,hAlign='RIGHT'))
            

        self.count += 1 
        # self.pdf_canvas.drawImage(imgdata, 60, self.height, width=500, height=500)
        # self.pdf_canvas.
        
    def save_pdf(self):
        # for image,name in zip(self.images,self.imnames):
        #     print(type(image))
        #     self.pdf_canvas.drawString(100, 100, name)
        #     self.pdf_canvas.drawImage(image, 30, 600, width=100, height=100)
        # self.pdf_canvas.save()
        self.flowables.append(Table(self.table))
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

    def standard_plots(self):
        train_path = str(self.folder.joinpath('Train'))
        test_path = str(self.folder.joinpath('Test'))
        eval_a_path = str(self.folder.joinpath('Eval_A'))
        eval_b_path = str(self.folder.joinpath('Eval_B'))
        print(test_path)
        #episode rewards
        print('starting standard plots')
        self.backend.draw_net_reward(test_path)
        self.plot_to_png('Net Reward')
        #ending distance
        self.backend.draw_shortest_goal_dist(test_path)
        self.plot_to_png('Average Ending Distance')
        #eval ending distance for hand A
        self.backend.draw_scatter_end_dist(eval_a_path)
        self.plot_to_png('Hand A Trained Ending Distance')
        #eval ending distance for hand B
        self.backend.draw_scatter_end_dist(eval_b_path)
        self.plot_to_png('Hand B Trained Ending Distance')
    
        print('saving to pdf')
        self.save_pdf()
        # self.make_gifs()

    def eval_plots(self):
        # this one makes all the ending distance plots and records the mean and std dev of that and the efficiency to a csv file while doing it
        suffixes = ["B","A",
         "2v2_70.30_70.30_1.1_53",
         "2v2_35.65_35.65_1.1_53",
         "2v2_65.35_65.35_1.1_63",
         "2v2_50.50_50.50_1.1_63",
         "2v2_70.30_70.30_1.1_63",
         "2v2_35.65_35.65_1.1_63",
         "2v2_65.35_65.35_1.1_73",
         "2v2_50.50_50.50_1.1_73",
         "2v2_70.30_70.30_1.1_73",
         "2v2_35.65_35.65_1.1_73"]
        csv_lists = [self.folder.stem]
        top_row = ["2v2_65.35_65.35_1.1_53",'','','',
         "2v2_50.50_50.50_1.1_53",'','','',
         "2v2_70.30_70.30_1.1_53",'','','',
         "2v2_35.65_35.65_1.1_53",'','','',
         "2v2_65.35_65.35_1.1_63",'','','',
         "2v2_50.50_50.50_1.1_63",'','','',
         "2v2_70.30_70.30_1.1_63",'','','',
         "2v2_35.65_35.65_1.1_63",'','','',
         "2v2_65.35_65.35_1.1_73",'','','',
         "2v2_50.50_50.50_1.1_73",'','','',
         "2v2_70.30_70.30_1.1_73",'','','',
         "2v2_35.65_35.65_1.1_73",'','','']
        middle_row = []
        for suf in suffixes:
            eval_path = str(self.folder.joinpath('Eval_'+suf))
            dist_nums = self.backend.draw_scatter_end_dist(eval_path)
            self.plot_to_png(suf)
            efficiency_nums = self.backend.draw_average_efficiency(eval_path)
            csv_lists.extend(dist_nums)
            csv_lists.extend(efficiency_nums)
            middle_row.extend(['Average Goal Distance','STD Dev','Average Efficiency','STD Dev'])
        self.save_pdf()
        
        with open('./distance_and_efficiency.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(csv_lists)

if __name__ == '__main__':
    base_path = pathlib.Path(__file__)
    base_path = base_path.parent
    
    base_path = base_path.joinpath('demos/rl_demo/data/HPC Runs/')
    base_path=str(base_path)+'/'
    subpaths = ['FTP_euler_3', 'FTP_euler_5', 'FTP_quat_3','FTP_quat_5','JA_euler_3','JA_euler_5','JA_quat_3','JA_quat_5']
    image_boi = ImageMaker(base_path+'FTP_euler_3',land=True)
    for path in subpaths:
        image_boi.backend.draw_orientation_success_rate(base_path+path+'/Eval_A/',5000)
        image_boi.plot_to_png(path)
    image_boi.thing=True
    # image_boi.count=0
    for path in subpaths:
        image_boi.backend.draw_end_poses(base_path+path+'/Eval_A/')
        image_boi.plot_to_png(path)
    image_boi.save_pdf()
