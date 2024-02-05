from mojograsp.simcore.data_gui_backend import PlotBackend
import numpy as np
import matplotlib.pyplot as plt
import PySimpleGUI as sg



class FigureGui():
    def __init__(self):
        data_layout =  [ [sg.Text('Hands to include on radar plot')],
                         [sg.Checkbox('FTP_Sim_S1_A_A',key='FTP_Sim_S1_A_A',default=False),sg.Checkbox('FTP_Sim_S2_A_A',key='FTP_Sim_S2_A_A',default=False),sg.Checkbox('FTP_Sim_S3_A_A',key='FTP_Sim_S3_A_A',default=False)],
                         [sg.Checkbox('FTP_Sim_S1_A_B',key='FTP_Sim_S1_A_B', default=False),sg.Checkbox('FTP_Sim_S2_A_B',key='FTP_Sim_S2_A_B', default=False),sg.Checkbox('FTP_Sim_S3_A_B',key='FTP_Sim_S3_A_B', default=False)],
                         [sg.Checkbox('FTP_Real_S1_A_A',key='FTP_Real_S1_A_A', default=False),sg.Checkbox('FTP_Real_S2_A_A',key='FTP_Real_S2_A_A', default=False),sg.Checkbox('FTP_Real_S3_A_A',key='FTP_Real_S3_A_A', default=False)],
                         [sg.Checkbox('FTP_Real_S1_A_B',key='FTP_Real_S1_A_B', default=False),sg.Checkbox('FTP_Real_S2_A_B',key='FTP_Real_S2_A_B', default=False),sg.Checkbox('FTP_Real_S3_A_B',key='FTP_Real_S3_A_B', default=False)],
                         [sg.Checkbox('JA_Sim_S1_A_A',key='JA_Sim_S1_A_A', default=False),sg.Checkbox('JA_Sim_S2_A_A',key='JA_Sim_S2_A_A', default=False),sg.Checkbox('JA_Sim_S3_A_A',key='JA_Sim_S3_A_A', default=False)],
                         [sg.Checkbox('JA_Sim_S1_A_B',key='JA_Sim_S1_A_B', default=False),sg.Checkbox('JA_Sim_S2_A_B',key='JA_Sim_S2_A_B', default=False),sg.Checkbox('JA_Sim_S3_A_B',key='JA_Sim_S3_A_B', default=False)],
                         [sg.Checkbox('JA_Real_S1_A_A',key='JA_Real_S1_A_A', default=False),sg.Checkbox('JA_Real_S2_A_A',key='JA_Real_S2_A_A', default=False),sg.Checkbox('JA_Real_S3_A_A',key='JA_Real_S3_A_A', default=False)],
                         [sg.Button("Go Avanced")]]
        self.high_level_folders = ['FTP_halfstate_A_rand','FTP_fullstate_A_rand',
                                   'FTP_state_3_old','JA_halfstate_A_rand',
                                   'JA_fullstate_A_rand','JA_state_3_old',
                                   'FTP_halfstate_B_rand','FTP_fullstate_B_rand',
                                   'FTP_state_3_B_old', 'JA_halfstate_B_rand',
                                   'JA_fullstate_B_rand','JA_state_3_B_old']
        self.low_level_folders = ['ast_a','ast_b','Real_A','Real_B']

        data_layout2 = [[sg.Checkbox('States',key='State',default=False), sg.Checkbox('Actions',key='Action',default=False),sg.Checkbox('Hands',key='Hand',default=False),sg.Checkbox('Domains',key='Domain',default=False)],
                        [sg.Button("Go")]]
        layout = [[sg.TabGroup([[sg.Tab('Simple Plotting', data_layout2)],
                        [sg.Tab('Advanced Plotting', data_layout)]])]]
        self.window = sg.Window('Analysis Window', layout, return_keyboard_events=True, use_default_focus=False, finalize=True)
        self.window.move(1000, 20)
        self.backend = PlotBackend('./demos/rl_demo/data/test')
        self.fig,_ = self.backend.get_figure()
        self.base_path = '/home/mothra/mojo-grasp/demos/rl_demo/data/'
        self.count = 0

    def tick(self):
        event, values = self.window.read()
        linstyle = ['solid','dotted','dashed']
        if event == 'Go':
            key = [values['State'],values['Action'],values['Hand'],values['Domain']]
            combinations = [[],[]]
            if key[0] and key[1]:
                highs = [self.high_level_folders[0],self.high_level_folders[1],self.high_level_folders[2],self.high_level_folders[3],self.high_level_folders[4],self.high_level_folders[5]]
            elif key[0] and not key[1]:
                highs = [[self.high_level_folders[0],self.high_level_folders[1],self.high_level_folders[2]],
                         [self.high_level_folders[3],self.high_level_folders[4],self.high_level_folders[5]],
                         [self.high_level_folders[6],self.high_level_folders[7],self.high_level_folders[8]],
                         [self.high_level_folders[9],self.high_level_folders[10],self.high_level_folders[11]]]
            elif not key[0] and key[1]:
                highs = [[self.high_level_folders[0],self.high_level_folders[3]],
                         [self.high_level_folders[1],self.high_level_folders[4]],
                         [self.high_level_folders[2],self.high_level_folders[5]]]
            else:
                highs = [[self.high_level_folders[0]],[self.high_level_folders[1]],[self.high_level_folders[2]],[self.high_level_folders[3]],[self.high_level_folders[4]],[self.high_level_folders[5]]]
            if key[2] and key[3]:
                lows = self.low_level_folders
            elif key[2] and not key[3]:
                lows = [[self.low_level_folders[0],self.low_level_folders[1]],[self.low_level_folders[2],self.low_level_folders[3]]]
            elif not key[2] and key[3]:
                lows = [[self.low_level_folders[0],self.low_level_folders[2]],[self.low_level_folders[1],self.low_level_folders[3]]]
            else:
                lows = [[self.low_level_folders[0]],[self.low_level_folders[1]],[self.low_level_folders[2]],[self.low_level_folders[3]]]
            for pre in highs:
                for post in lows:
                    # print(f"pre {pre}, post {post}")
                    if (len(pre) == 1) and (len(post) == 1):
                        path_folders = [pre[0] + '/' + post[0]]
                    elif (len(pre) > 1) and (len(post) == 1):
                        path_folders = [p + '/' + post[0] for p in pre]
                    elif (len(pre) == 1) and (len(post) > 1):
                        path_folders = [pre[0] + '/' + p for p in post]
                    elif (len(pre) > 1) and (len(post) > 1):
                        path_folders = [p + '/' + p2 for p in pre for p2 in post]
                    try:
                        [self.backend.draw_radar(self.base_path + fold, fold, linstyle[i]) for i,fold in enumerate(path_folders)] 
                        self.plot_to_png(pre)
                    except:
                        pass
                    self.backend.clear_axes()
                    self.backend.legend=[]
        elif event == 'Go Advanced':
            key = []

    def close(self):
        self.window.close()

    def plot_to_png(self,fig_name=''):
        self.fig.canvas.draw()
        
        plt.savefig(f'{fig_name}_{self.count}.png')
        self.count +=1
        return True

if __name__ == '__main__':
    thing = FigureGui()

    while True:
        thing.tick()
    thing.close()