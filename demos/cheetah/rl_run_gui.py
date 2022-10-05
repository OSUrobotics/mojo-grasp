#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 12:28:33 2022

@author: orochi
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 14:21:38 2022

@author: Nigel Swenson
"""

import PySimpleGUI as sg
import os
import pickle as pkl
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
import numpy as np
import json
from PIL import ImageGrab
from scipy.stats import kde
import csv
from AppleClassifier import AppleClassifier
import torch
import numpy as np

import argparse
from Ablation import perform_ablation
from torch.utils.data import TensorDataset, DataLoader
from utils import RNNDataset

import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle as pkl
import time
from csv_process import GraspProcessor
import argparse
from AppleClassifier import AppleClassifier
# from itertools import islice
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import datetime
from copy import deepcopy
import threading

'''
    Data Plotter
    
    This is based on the Demo_PNG_Viewer by PySimpleGUI
    
'''

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

class RNNGui():
    def __init__(self):
        self.toggle_btn_off = b'iVBORw0KGgoAAAANSUhEUgAAACgAAAAoCAYAAACM/rhtAAAABmJLR0QA/wD/AP+gvaeTAAAED0lEQVRYCe1WTWwbRRR+M/vnv9hO7BjHpElMKSlpqBp6gRNHxAFVcKM3qgohQSqoqhQ45YAILUUVDRxAor2VAweohMSBG5ciodJUSVqa/iikaePEP4nj2Ovdnd1l3qqJksZGXscVPaylt7Oe/d6bb9/svO8BeD8vA14GvAx4GXiiM0DqsXv3xBcJU5IO+RXpLQvs5yzTijBmhurh3cyLorBGBVokQG9qVe0HgwiXLowdy9aKsY3g8PA5xYiQEUrsk93JTtjd1x3siIZBkSWQudUK4nZO1w3QuOWXV+HuP/fL85klAJuMCUX7zPj4MW1zvC0Ej4yMp/w++K2rM9b70sHBYCjo34x9bPelsgp/XJksZ7KFuwZjr3732YcL64ttEDw6cq5bVuCvgy/sje7rT0sI8PtkSHSEIRIKgCQKOAUGM6G4VoGlwiqoVd2Za9Vl8u87bGJqpqBqZOj86eEHGNch+M7otwHJNq4NDexJD+59RiCEQG8qzslFgN8ibpvZNsBifgXmFvJg459tiOYmOElzYvr2bbmkD509e1ylGEZk1Y+Ssfan18n1p7vgqVh9cuiDxJPxKPT3dfGXcN4Tp3dsg/27hUQs0qMGpRMYjLz38dcxS7Dm3nztlUAb38p0d4JnLozPGrbFfBFm79c8hA3H2AxcXSvDz7/+XtZE1kMN23hjV7LTRnKBh9/cZnAj94mOCOD32gi2EUw4FIRUMm6LGhyiik86nO5NBdGRpxYH14bbjYfJteN/OKR7UiFZVg5T27QHYu0RBxoONV9W8KQ7QVp0iXdE8fANUGZa0QAvfhhXlkQcmjJZbt631oIBnwKmacYoEJvwiuFgWncWnXAtuVBBEAoVVXWCaQZzxmYuut68b631KmoVBEHMUUrJjQLXRAQVSxUcmrKVHfjWWjC3XOT1FW5QrWpc5IJdQhDKVzOigEqS5dKHMVplnNOqrmsXqUSkn+YzWaHE9RW1FeXL7SKZXBFUrXW6jIV6YTEvMAUu0W/G3kcxPXP5ylQZs4fa6marcWvvZfJu36kuHjlc/nMSuXz+/ejxgqPFpuQ/xVude9eu39Jxu27OLvBGoMjrUN04zrNMbgVmOBZ96iPdPZmYntH5Ls76KuxL9NyoLA/brav7n382emDfHqeooXyhQmARVhSnAwNNMx5bu3V1+habun5nWdXhwJZ2C5mirTesyUR738sv7g88UQ0rEkTDlp+1wwe8Pf0klegUenYlgyg7bby75jUTITs2rhCAXXQ2vwxz84vlB0tZ0wL4NEcLX/04OrrltG1s8aOrHhk51SaK0us+n/K2xexBxljcsm1n6x/Fuv1PCWGiKOaoQCY1Vb9gWPov50+fdEqd21ge3suAlwEvA14G/ucM/AuppqNllLGPKwAAAABJRU5ErkJggg=='
        self.toggle_btn_on = b'iVBORw0KGgoAAAANSUhEUgAAACgAAAAoCAYAAACM/rhtAAAABmJLR0QA/wD/AP+gvaeTAAAD+UlEQVRYCe1XzW8bVRCffbvrtbP+2NhOD7GzLm1VoZaPhvwDnKBUKlVyqAQ3/gAkDlWgPeVQEUCtEOIP4AaHSI0CqBWCQyXOdQuRaEFOk3g3IMWO46+tvZ+PeZs6apq4ipON1MNafrvreTPzfvub92bGAOEnZCBkIGQgZOClZoDrh25y5pdjruleEiX+A+rCaQo05bpuvJ/+IHJCSJtwpAHA/e269g8W5RbuzF6o7OVjF8D3Pr4tSSkyjcqfptPDMDKSleW4DKIggIAD5Yf+Oo4DNg6jbUBlvWLUNutAwZu1GnDjzrcXzGcX2AHw/emFUV6Sfk0pqcKpEydkKSo9q3tkz91uF5aWlo1Gs/mYc+i7tz4//19vsW2AU9O381TiioVCQcnlRsWeQhD3bJyH1/MiFLICyBHiuzQsD1arDvypW7DR9nzZmq47q2W95prm+I9fXfqXCX2AF2d+GhI98Y8xVX0lnxvl2UQQg0csb78ag3NjEeD8lXZ7pRTgftmCu4864OGzrq+5ZU0rCa3m+NzXlzvoAoB3+M+SyWQuaHBTEzKMq/3BMbgM+FuFCDBd9kK5XI5PJBKqLSev+POTV29lKB8rT0yMD0WjUSYLZLxzNgZvIHODOHuATP72Vwc6nQ4Uiw8MUeBU4nHS5HA6TYMEl02wPRcZBJuv+ya+UCZOIBaLwfCwQi1Mc4QXhA+PjWRkXyOgC1uIhW5Qd8yG2TK7kSweLcRGKKVnMNExWWBDTQsH9qVmtmzjiThQDs4Qz/OUSGTwcLwIQTLW58i+yOjpXDLqn1tgmDzXzRCk9eDenjo9yhvBmlizrB3V5dDrNTuY0A7opdndStqmaQLPC1WCGfShYRgHdLe32UrV3ntiH9LliuNrsToNlD4kruN8v75eafnSgC6Luo2+B3fGKskilj5muV6pNhk2Qqg5v7lZ51nBZhNBjGrbxfI1+La5t2JCzfD8RF1HTBGJXyDzs1MblONulEqPDVYXgwDIfNx91IUVbAbY837GMur+/k/XZ75UWmJ77ou5mfM1/0x7vP1ls9XQdF2z9uNsPzosXPNFA5m0/EX72TBSiqsWzN8z/GZB08pWq9VeEZ+0bjKb7RTD2i1P4u6r+bwypo5tZUumEcDAmuC3W8ezIqSGfE6g/sTd1W5p5bKjaWubrmWd29Fu9TD0GlYlmTx+8tTJoZeqYe2BZC1/JEU+wQR5TVEUPptJy3Fs+Vkzgf8lemqHumP1AnYoMZSwsVEz6o26i/G9Lgitb+ZmLu/YZtshfn5FZDPBCcJFQRQ+8ih9DctOFvdLIKHH6uUQnq9yhFu0bec7znZ+xpAGmuqef5/wd8hAyEDIQMjAETHwP7nQl2WnYk4yAAAAAElFTkSuQmCC'
        self.data_dict = {'train': {'state':[],'label':[]}, 'validation':{'state':[],'label':[]}, 'test': {'state':[],'label':[]}}
        # define menu layout
        self.menu = [['File', ['Open Folder', 'Exit']], ['Help', ['About', ]]]
        self.args = {}
        self.shuffle_type = 'Episode'
        self.model_path = '/'
        self.train_path = '/'
        self.validation_path = '/'
        self.test_path = '/'
        self.Classifier = None
        self.train_dataset, self.validation_dataset = None, None
        
#        plot_buttons = [[sg.Button('Object Path', size=(8, 2)), sg.Button('Finger Angles', size=(8, 2))],
#                        [sg.Button('Actor Output', size=(8, 2)), sg.Button('Critic Output', size=(8, 2)), sg.Button('Rewards', size=(8, 2))],
#                        [sg.Button('Explored Region', size=(8,2))],
#                        [sg.Text("Keep previous graph", size=(10, 3), key='-toggletext-'), sg.Button(image_data=toggle_btn_off, key='-TOGGLE--', button_color=(sg.theme_background_color(), sg.theme_background_color()), border_width=0, metadata=False)]]
        # define layout, show and read the window
        
        data_layout =  [ [sg.Text('Path to Training Data'), sg.Text("Make Validation Set from Training Data"),sg.Button(image_data=self.toggle_btn_off, key='-Split-', button_color=(sg.theme_background_color(), sg.theme_background_color()), border_width=0, metadata=False)],
                         [sg.Button("Browse",key='browse-training',button_color='DarkBlue'),sg.Text("/", key='training-path')],
                         [sg.Text('Path to Validation Data')],
                         [sg.Button("Browse",key='browse-validation',button_color='DarkBlue'),sg.Text("/", key='validation-path')],
                         [sg.Text('Path to Testing Data')],
                         [sg.Button("Browse",key='browse-testing',button_color='DarkBlue'),sg.Text("/", key='testing-path')],
                         [sg.Text("Shuffle Type")],
                         [sg.OptionMenu(values=('Episode', 'Timestep', 'None'),  k='shuffle-type', default_value='Episode')]]
        
        model_layout = [ [sg.Text('Num Epochs'), sg.Input(1000, key='-epochs')],
                         [sg.Text('Shape'),sg.Input('cube')],
                         [sg.Text('Reward Setup'), sg.Checkbox('Distance', default=False, k='-dist-'), sg.Checkbox('Contact', default=False, k='-contact-'), sg.Checkbox('End Penalty', default=False, k='-end-'), sg.Checkbox('Sparse', default=False, k='-sparse-')],
                         [sg.Button('Load Model?', size=(8, 2), key='-load-model')],
                         [sg.Text("/", key='-model-path')]]
        
        plotting_layout = [[sg.Text('Model Title')],
                       [sg.Input(key='-Title-')],
                       [sg.Text("Graphs")],
                       [sg.Checkbox('Accuracy', default=False, k='-acc-')],
                       [sg.Checkbox('AUC', default=False, k='-auc-')],
                       [sg.Checkbox('Loss', default=False, k='-loss-')],
                       [sg.Checkbox('TP/FP', default=False, k='-TPFP-')],
                       [sg.Checkbox('Example', default=False, k='-ex-')],
                       [sg.Button('Begin Training', key='-train', bind_return_key=True)],
                       [sg.Text('Work progress'), sg.ProgressBar(100, size=(20, 20), orientation='h', key='-PROG-')]]

        layout = [[sg.TabGroup([[sg.Tab('Data', data_layout, key='-mykey-'),
                                sg.Tab('Model', model_layout),
                                sg.Tab('Plotting', plotting_layout)]], key='-group1-', tab_location='top', selected_title_color='purple')]]
            
        self.data_type = None
        self.window = sg.Window('RNN Gui', layout, return_keyboard_events=True, use_default_focus=False, finalize=True)
        
    def load_pkl(self, filename, filenum, datatype='train'):
        with open(filename, 'rb') as pkl_file:
            pkl_dict = pkl.load(pkl_file)
            self.data_dict[datatype]['state'].append(pkl_dict['state'])
            self.data_dict[datatype]['label'].append(pkl_dict['label'])
            self.data_dict[datatype]['title'].append(filenum)
    
    def load_csv(self, filename, filenum, datatype='train'):
        with open(filename, newline='') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            state = []
            label = []
            for row in csv_reader:
                state.append(row[:-1])
                label.append(row[-1])
            self.data_dict[datatype]['state'].append(state)
            self.data_dict[datatype]['label'].append(label)
            self.data_dict[datatype]['title'].append(filenum)
             
    def clear_data(self):
        self.data_dict = {'train': {'state':[],'label':[], 'title':[]}, 'validation':{'state':[],'label':[], 'title':[]}, 'test': {'state':[],'label':[], 'title':[]}}
        
    def load_data(self, directory_path, datatype='train'):
        self.clear_data()
        csv_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.lower().endswith('.csv')]
        if len(csv_files) > 0:
            for i, filename in enumerate(csv_files):
                self.load_csv(filename, i, datatype)
        else:
            pkl_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.lower().endswith('.pkl')]
            if len(pkl_files) > 0:
                for i, filename in enumerate(pkl_files):
                    self.load_pkl(filename, i, datatype)
            else:
                sg.popup('No csv or pkl files in this folder')

    def build_dataset(self):
        self.train_dataset = RNNDataset(self.data_dict['train']['state'], self.data_dict['train']['label'],
                                        self.data_dict['train']['title'], self.args['batch_size'], range_params=False)
        params = self.train_dataset.get_params()

        self.validation_dataset = RNNDataset(self.data_dict['validation']['state'], self.data_dict['validation']['label'],
                                       self.data_dict['validation']['title'], self.args['batch_size'], range_params=False)
        np.save('proxy_mins_and_maxs', params)

    def build_args(self):
        self.args = {'epochs':self.window['-epochs'],
                     'batch_size':self.window['-batch-size'],
                     'hiddens':self.window['-nodes'],
                     'layers':self.window['-layers'],
                     'model_type':self.window['-model'],
                     'drop_prob':self.window['-drop'],
                     'reduced':False,
                     'goal':'grasp',
                     's_f_bal': None,
                     'phase':'full',
                     'evaluate':False}
        
    def make_and_train(self):
        self.build_args()
        self.build_dataset()
        self.Classifier = AppleClassifier(self.train_dataset, self.validation_dataset, self.args)
        self.Classifier.train(self.window)
        self.Classifier.save_metadata()
        print('model finished, saving now')
        self.Classifier.save_data()
        self.Classifier.save_model()

    def run_gui(self):
        while True:

            event, values = self.window.read()
            # --------------------- Button & Keyboard ---------------------
            if event == sg.WIN_CLOSED:
                break
            elif event == 'shuffle-type':
                self.shuffle_type = values['shuffle-type']
            elif event == '-load-model':
                newfolder = sg.popup_get_file('Select Model File', no_window=True)
                if newfolder is None:
                    continue
                if newfolder.lower().endswith('.pt'):
                    self.model_path = newfolder
            elif event == '-Split-':  # if the graphical button that changes images
                self.window['-Split-'].metadata = not self.window['-Split-'].metadata
                self.window['-Split-'].update(image_data=self.toggle_btn_on if self.window['-Split-'].metadata else self.toggle_btn_off)
                if self.window['-Split-'].metadata:
                    self.window['browse-validation'].update(button_color='gray')
                else:
                    self.window['browse-validation'].update(button_color='DarkBlue')
            elif event == 'Exit':
                break
            # ----------------- Menu choices -----------------
            if event == 'browse-training':
                newfolder = sg.popup_get_folder('Select Training Folder', no_window=True)
                if newfolder is None:
                    continue
    
                folder = newfolder
                self.load_data(folder)
    
#                self.window['training-path'].update(values=folder)
                self.window.refresh()
            
            elif (event == 'browse-validation') & (not self.window['-Split-'].metadata):
                newfolder = sg.popup_get_folder('Select Validation Folder', no_window=True)
                if newfolder is None:
                    continue
    
                folder = newfolder
                self.load_data(folder)
    
#                self.window['validation-path'].update(folder)
                self.window.refresh()
                
            elif event == 'browse-testing':
                newfolder = sg.popup_get_folder('Select Testing Folder', no_window=True)
                if newfolder is None:
                    continue
    
                folder = newfolder
                self.load_data(folder)
    
#                self.window['testing-path'].update(folder)
            elif event == '-train':
                thread = threading.Thread(target=self.make_and_train, daemon=True)
                thread.start()
            
    
            elif event == 'About':
                sg.popup('Why you click me?',
                         'Go harrass Nigel with questions. swensoni@oregonstate.edu')
            self.window['-model-path'].update(self.model_path)
            self.window['training-path'].update(self.train_path)
            self.window['validation-path'].update(self.validation_path)
            self.window['testing-path'].update(self.model_path)
        self.window.close()
        

def main():


    backend = RNNGui()
    backend.run_gui()
#    # loop reading the user input and displaying image, filename
#    while True:
#
#        event, values = window.read()
#        # --------------------- Button & Keyboard ---------------------
#        if event == sg.WIN_CLOSED:
#            break
#        elif event in ('MouseWheel:Down', 'Down:40', 'Next:34') and filenum < len(episode_files)-1:
#            filenum += 1
#            filename = os.path.join(folder, filenames_only[filenum])
#            window['-LISTBOX-'].update(set_to_index=filenum, scroll_to_index=filenum)
#            backend.load_data(filename)
#        elif event in ('MouseWheel:Up', 'Up:38', 'Prior:33') and filenum > 0:
#            filenum -= 1
#            filename = os.path.join(folder, filenames_only[filenum])
#            window['-LISTBOX-'].update(set_to_index=filenum, scroll_to_index=filenum)
#            backend.load_data(filename)
#        elif event == 'Exit':
#            break
#        elif event == '-LISTBOX-':
#            filename = os.path.join(folder, values['-LISTBOX-'][0])
#            filenum = episode_files.index(filename)
#            backend.load_data(filename)
#        elif event == 'Object Path':
#            backend.draw_path()
#        elif event == 'Finger Angles':
#            backend.draw_angles()
#        elif event == 'Actor Output':
#            backend.draw_actor_output()
#        elif event == 'Critic Output':
#            backend.draw_critic_output()
#        elif event == 'Rewards':
#            backend.draw_rewards()
#        elif event == 'Explored Region':
#            backend.draw_explored_region()
#        elif event == '-TOGGLE-GRAPHIC-':  # if the graphical button that changes images
#            window['-TOGGLE-GRAPHIC-'].metadata = not window['-TOGGLE-GRAPHIC-'].metadata
#            window['-TOGGLE-GRAPHIC-'].update(image_data=toggle_btn_on if window['-TOGGLE-GRAPHIC-'].metadata else toggle_btn_off)
#            backend.clear_plots = not backend.clear_plots
#        elif event == '-SAVE-':
#            filename=r'test.png'
#            save_element_as_file(window['-CANVAS-'], filename)
#        # ----------------- Menu choices -----------------
#        if event == 'Open Folder':
#            newfolder = sg.popup_get_folder('New folder', no_window=True)
#            if newfolder is None:
#                continue
#
#            folder = newfolder
#            episode_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith('.pkl')]
#            filenames_only = [f for f in os.listdir(folder) if f.lower().endswith('.pkl')]
#
#            window['-LISTBOX-'].update(values=filenames_only)
#            window.refresh()
#
#            filenum = 0
#        elif event == 'About':
#            sg.popup('Demo pkl Viewer Program',
#                     'Please give PySimpleGUI a try!')
#
#        # update window with new image
##        window['-IMAGE-'].update(filename=filename)
#        # update window with filename
#        window['-FILENAME-'].update(filename)
#        # update page display
#        window['-FILENUM-'].update('File {} of {}'.format(filenum + 1, len(episode_files)))
#
#    window.close()

if __name__ == '__main__':
    main()