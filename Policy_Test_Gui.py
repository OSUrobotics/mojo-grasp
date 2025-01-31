#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 14:21:38 2022

@author: Nigel Swenson
"""

import PySimpleGUI as sg
import os
import pickle as pkl
import numpy as np
import json
from PIL import ImageGrab


# from itertools import islice
import threading
from mojograsp.simcore.run_from_file import run_pybullet
import pathlib
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
        self.save_path = '/'
        self.expert_path = '/'
        self.load_path  = '/'
        self.built = False
        self.train_dataset, self.validation_dataset = None, None
        
        # define layout, show and read the window
        data_layout =  [ [sg.Text('Model Type'), sg.OptionMenu(values=('DDPG', 'DDPGFD','DDPG+HER', 'DDPGFD+HER'),  k='-model', default_value='DDPG')],
                         [sg.Text('Path to Expert Data if using FD')],
                         [sg.Button("Browse",key='-browse-expert',button_color='DarkBlue'),sg.Text("/", key='-expert-path')],
                         [sg.Text('Path to Save Data')],
                         [sg.Button("Browse",key='-browse-save',button_color='DarkBlue'),sg.Text("/", key='-save-path')],
                         [sg.Text('Path to Previous Policy if Transferring')],
                         [sg.Button("Browse",key='-browse-load',button_color='DarkBlue'),sg.Text("/", key='-load-path')],
                         [sg.Text('Object'), sg.OptionMenu(values=('Cube', 'Cylinder'), k='-object', default_value='Cube')],
                         [sg.Text('Hand'), sg.OptionMenu(values=('2v2', '2v2-B'), k='-hand', default_value='2v2')],
                         [sg.Text("Task"), sg.OptionMenu(values=('asterisk','random','full_random'), k='-task', default_value='full_random')],
                         [sg.Text('Replay Buffer Sampling'), sg.OptionMenu(values=('priority','random','random+expert'), k='-sampling', default_value='priority')]]
        
        
        model_layout = [ [sg.Text('Num Epochs'), sg.Input(10000, key='-epochs'), sg.Text('Batch Size'), sg.Input(100, key='-batch-size')],
                         [sg.Text('Learning Rate'), sg.Input(0.0001,key='-learning'), sg.Text('Discount Factor'), sg.Input(0.995, key='-df')],
                         [sg.Text('Starting Epsilon'), sg.Input(0.7,key='-epsilon'), sg.Text('Epsilon Decay Rate'), sg.Input(0.998, key='-edecay')],
                         [sg.Text('Rollout Size'), sg.Input(5,key='-rollout_size'), sg.Text('Rollout Weight'), sg.Input(0.5, key='-rollout_weight')],
                         [sg.Text('Evaluation Period'), sg.Input(3,key='-eval'), sg.Text('Tau'), sg.Input(0.0005, key='-tau')],
                         [sg.Text('Timesteps per Episode'), sg.Input(150,key='-tsteps'), sg.Text('Timesteps in Evaluation'), sg.Input(150,key='-eval-tsteps')],
                         [sg.Text('State Training Noise'), sg.Input(0.05, key='-snoise'),sg.Text('Start Pos Range (mm)'), sg.Input(0, key='-start-noise')]]
        
        plotting_layout = [[sg.Text('Model Title')],
                       [sg.Input('test1',key='-title')],
                       [sg.Text("State")],
                       [sg.Checkbox('Finger Tip Position', default=True, k='-ftp')],
                       [sg.Checkbox('Finger Base Position', default=False, k='-fbp')],
                       [sg.Checkbox('Finger Contact Position', default=False, k='-fcp')],
                       [sg.Checkbox('Joint Angle', default=False, k='-ja')],
                       [sg.Checkbox('Object Position', default=True, k='-op')],
                       [sg.Checkbox('Finger Object Distance', default=False, k='-fod')],
                       [sg.Checkbox('Finger Tip Angle',default=True, k='-fta')],
                       [sg.Checkbox('Goal Position',default=True, k='-gp')],
                       [sg.Text('Num Previous States'),sg.Input('0', k='-pv')],
                       [sg.Text("Reward"), sg.OptionMenu(values=('Sparse','Distance','Distance + Finger', 'Hinge Distance + Finger', 'Slope', 'Slope + Finger'), k='-reward',default_value='Distance + Finger'), sg.Text('Success Radius (mm)'), sg.Input(2, key='-sr'),],
                       [sg.Text("Distance Scale"),  sg.Input(1,key='-distance_scale'), sg.Text('Contact Scale'),  sg.Input(0.2,key='-contact_scale')],
                       [sg.Text("Action"), sg.OptionMenu(values=('Joint Velocity','Finger Tip Position'), k='-action',default_value='Finger Tip Position')],
                       [sg.Checkbox('Vizualize Simulation',default=False, k='-viz'), sg.Checkbox('Real World?',default=False, k='-rw')],
                       [sg.Button('Begin Training', key='-train', bind_return_key=True)],
                       [sg.Button('Build Config File WITHOUT Training', key='-build')],
                       [sg.Text('Work progress'), sg.ProgressBar(100, size=(20, 20), orientation='h', key='-PROG-')]]

        layout = [[sg.Menu(menu)], [sg.(col)]]
            
        self.data_type = None
        self.window = sg.Window('RNN Gui', layout, return_keyboard_events=True, use_default_focus=False, finalize=True)

    def build_args(self, values):
        RW = bool(values['-rw'])
        self.built = False
        print('building RL arglist, real world setting: ',RW)
        self.args = {'epochs': int(values['-epochs']),
                     'batch_size': int(values['-batch-size']),
                     'model': values['-model'],
                     'learning_rate': float(values['-learning']),
                     'discount': float(values['-df']),
                     'epsilon': float(values['-epsilon']),
                     'edecay': float(values['-edecay']),
                     'object': values['-object'],
                     'hand': values['-hand'],
                     'task': values['-task'],
                     'evaluate': int(values['-eval']),
                     'sampling': values['-sampling'],
                     'reward': values['-reward'],
                     'action': values['-action'],
                     'rollout_size': int(values['-rollout_size']),
                     'rollout_weight': float(values['-rollout_weight']),
                     'tau': float(values['-tau']),
                     'pv': int(values['-pv']),
                     'viz': int(values['-viz']),
                     'sr': int(values['-sr']),
                     'state_noise': float(values['-snoise']),
                     'start_noise': float(values['-start-noise']),
                     'tsteps': int(values['-tsteps']),
                     'eval-tsteps':int(values['-eval-tsteps']),
                     'distance_scaling': float(values['-distance_scale']),
                     'contact_scaling': float(values['-contact_scale'])}
        state_len = 0
        state_mins = []
        state_maxes = []
        state_list = []
        if values['-ftp']:
            if not RW:
                state_mins.extend([-0.072, 0.018, -0.072, 0.018])
                state_maxes.extend([0.072, 0.172, 0.072, 0.172])
            elif RW:
                state_mins.extend([-0.108, 0.132, -0.108, 0.132])
                state_maxes.extend([0.108, 0.348, 0.108, 0.348])
            state_len += 4
            state_list.append('ftp')
        if values['-fbp']:
            if not RW:
                state_mins.extend([-0.072, 0.018, -0.072, 0.018])
                state_maxes.extend([0.072, 0.172, 0.072, 0.172])
            elif RW:
                state_mins.extend([-0.108, 0.132, -0.108, 0.132])
                state_maxes.extend([0.108, 0.348, 0.108, 0.348])
            state_len += 4
            state_list.append('fbp')
        if values['-fcp']:
            if not RW:
                state_mins.extend([-0.072, 0.018, -0.072, 0.018])
                state_maxes.extend([0.072, 0.172, 0.072, 0.172])
            elif RW:
                state_mins.extend([-0.108, 0.132, -0.108, 0.132])
                state_maxes.extend([0.108, 0.348, 0.108, 0.348])
            state_len += 4
            state_list.append('fcp')
        if values['-op']:
            if not RW:
                state_mins.extend([-0.072, 0.018])
                state_maxes.extend([0.072, 0.172])
            elif RW:
                state_mins.extend([-0.108, 0.132])
                state_maxes.extend([0.108, 0.348])
            state_len += 2
            state_list.append('op')
        if values['-ja']:
            state_mins.extend([-np.pi/2, -np.pi, -np.pi/2, 0])
            state_maxes.extend([np.pi/2, 0, np.pi/2, np.pi])
            state_len += 4
            state_list.append('ja')
        if values['-fod']:
            if not RW:
                state_mins.extend([-0.001, -0.001])
                state_maxes.extend([0.072, 0.072])
            elif RW:
                state_mins.extend([-0.001, -0.001])
                state_maxes.extend([0.108, 0.108])
            state_len += 2
            state_list.append('fod')
        if values['-fta']:
            state_mins.extend([-np.pi/2, -np.pi/2])
            state_maxes.extend([np.pi/2, np.pi/2])
            state_len += 2
            state_list.append('fta')
        if values['-gp']:
            if not RW:
                state_mins.extend([-0.07, -0.07])
                state_maxes.extend([0.07, 0.07])
            elif RW:
                state_mins.extend([-0.105, -0.105])
                state_maxes.extend([0.105, 0.105])
            state_len += 2
            state_list.append('gp')
        if self.args['pv'] > 0:
            state_len += state_len * self.args['pv']
            temp_mins = state_mins.copy()
            temp_maxes = state_maxes.copy()
            for i in range(self.args['pv']):
                state_mins.extend(temp_mins)
                state_maxes.extend(temp_maxes)
        if state_len == 0:
            print('No selected state space')
            return False
        if (self.args['task'] == 'asterisk') or (self.args['task'] == 'random'):
            if not values['-gp']:
                print('Goal position needed for multigoal tasks')
                return False
        self.args['state_dim'] = state_len
        self.args['state_mins'] = state_mins
        self.args['state_maxes'] = state_maxes
        self.args['state_list'] = state_list

        if self.args['action'] =='Joint Velocity' or self.args['action'] =='Finger Tip Position':
            self.args['action_dim'] = 4

        if 'FD' in self.args['model']:
            exists = os.path.isfile(self.expert_path + 'episode_all.pkl')
            if not exists:
                print('Selected FD model but no expert data loaded')
                return False
            else:
                self.args['edata'] = values['-browse-expert'] + 'episode_all.pkl'
        if os.path.isdir(self.save_path) and self.save_path != '/':
            self.args['save_path'] = self.save_path + '/'
            self.args['load_path'] = self.load_path + '/'
        else:
            print('save path is not a valid directory')
            return False
        overall_path = pathlib.Path(__file__).parent.resolve()
        resource_path = overall_path.joinpath('demos/rl_demo/resources')
        run_path = overall_path.joinpath('demos/rl_demo/runs')
        self.args['tname'] = str(run_path.joinpath(values['-title']))
        if values['-hand'] == '2v2':
            self.args['hand_path'] = str(resource_path.joinpath('2v2_Hand_A/hand/2v2_50.50_50.50_1.1_53.urdf'))
        elif values['-hand'] == '2v2-B':
            self.args['hand_path'] = str(resource_path.joinpath('2v2_Hand_B/hand/2v2_65.35_65.35_1.1_53.urdf'))
        if values['-object'] == 'Cube':
            self.args['object_path'] = str(resource_path.joinpath('object_models/2v2_mod/2v2_mod_cuboid_small.urdf'))
        elif values['-object'] == 'Cylinder':
            self.args['object_path'] = str(resource_path.joinpath('resources/object_models/2v2_mod/2v2_mod_cylinder_small_alt.urdf'))
        if values['-action'] == 'Joint Velocity':
            self.args['max_action'] = 1.57
        elif values['-action'] == 'Finger Tip Position':
            self.args['max_action'] = 0.01
        if values['-task'] == 'full_random':
            self.args['points_path'] = str(resource_path.joinpath('points.csv'))
        else:
            self.args['points_path'] = str(resource_path.joinpath('train_points.csv'))
        
        self.built = True
        

        return True
        
    def train(self):
        run_pybullet(self.args['save_path'] + 'experiment_config.json', self.window)
        print('model finished, saving now')

    def log_params(self):
        if self.built:
            print('saving configuration')
            
            with open(self.args['save_path'] + '/experiment_config.json', 'w') as conf_file:
                json.dump(self.args, conf_file, indent=4)
            try:
                os.mkdir(self.args['save_path'] + '/Train/')
            except FileExistsError:
                pass
            try:
                os.mkdir(self.args['save_path'] + '/Test/')
            except FileExistsError:
                pass
        else:
            print('config not built, parameters not saved')

    def run_gui(self):
        while True:

            event, values = self.window.read()
            # print(values.keys())
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

            elif event == 'Exit':
                break
            # ----------------- Menu choices -----------------
            if event == '-browse-expert':
                newfolder = sg.popup_get_folder('Select Folder Containing Expert Data', no_window=True)
                if newfolder is None:
                    continue
    
                folder = newfolder
                print(type(folder))
                self.expert_path = folder
    
                self.window.refresh()
            
            elif event == '-browse-save':
                newfolder = sg.popup_get_folder('Select Folder To Save Data In', no_window=True)
                if newfolder is None:
                    continue
    
                folder = newfolder
                print(type(folder))
                self.save_path = folder
    
                self.window.refresh()

            elif event == '-browse-load':
                newfolder = sg.popup_get_folder('Select Folder To Save Data In', no_window=True)
                if newfolder is None:
                    continue
    
                folder = newfolder
                print(type(folder))
                self.load_path = folder
    
                self.window.refresh()
                
            elif event == '-train':
                ready = self.build_args(values)
                if ready:
                    self.log_params()
                    thread = threading.Thread(target=self.train, daemon=True)
                    thread.start()
                else:
                    print('Parameters incorrect, cant start training')

            elif event == '-build':
                ready = self.build_args(values)        
                if ready:
                    self.log_params()
                    print('Build Successful')
                else:
                    print('Build Not Successful')
                
            # elif event == '-update':
            #     print(values['-update'])
                
            elif event == 'About':
                sg.popup('Why you click me?',
                         'Go harrass Nigel with questions. swensoni@oregonstate.edu')
            self.window['-save-path'].update(self.save_path)
            self.window['-expert-path'].update(self.expert_path)
            self.window['-load-path'].update(self.load_path)
        self.window.close()
        

def main():


    backend = RNNGui()
    backend.run_gui()


if __name__ == '__main__':
    main()