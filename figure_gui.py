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
                         [sg.Button("Go Avanced")] ]
        self.high_level_folders = ['Mothra_Slide/','Misc_Slide/','HPC_Slide/']
        self.mid_level = ['JA_S1', 'JA_S2', 'JA_S3','FTP_S1', 'FTP_S2', 'FTP_S3']
        self.low_level_folders = ['Ast_A','Ast_B','Real_A','Real_B']

        data_layout2 = [[sg.Checkbox('States',key='State',default=False), sg.Checkbox('Actions',key='Action',default=False),sg.Checkbox('Hands',key='Hand',default=False),sg.Checkbox('Domains',key='Domain',default=False)],
                        [sg.Button("Go")]]
        layout = [[sg.TabGroup([[sg.Tab('Simple Plotting', data_layout2)],
                        [sg.Tab('Advanced Plotting', data_layout)]])]]
        self.window = sg.Window('Analysis Window', layout, return_keyboard_events=True, use_default_focus=False, finalize=True)
        self.window.move(1000, 20)
        self.backend = PlotBackend('./demos/rl_demo/data/Mothra_Rotation/JA_S1')
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
                highs = [[self.mid_level[0],self.mid_level[1],self.mid_level[2],self.mid_level[3],self.mid_level[4],self.mid_level[5]]]
            elif key[0] and not key[1]:
                highs =  [[self.mid_level[0],self.mid_level[1],self.mid_level[2]],
                           [self.mid_level[3],self.mid_level[4],self.mid_level[5]]]
            elif not key[0] and key[1]:
                highs = [[self.mid_level[0],self.mid_level[3]],
                         [self.mid_level[1],self.mid_level[4]],
                         [self.mid_level[2],self.mid_level[5]]]
            else:
                highs = [[self.mid_level[0]],[self.mid_level[1]],[self.mid_level[2]],[self.mid_level[3]],[self.mid_level[4]],[self.mid_level[5]]]
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
                    print(f"pre {pre}, post {post}")
                    if (len(pre) == 1) and (len(post) == 1):
                        path_folders = [pre[0] + '/' + post[0]]
                    elif (len(pre) > 1) and (len(post) == 1):
                        path_folders = [p + '/' + post[0] for p in pre]
                    elif (len(pre) == 1) and (len(post) > 1):
                        path_folders = [pre[0] + '/' + p for p in post]
                    elif (len(pre) > 1) and (len(post) > 1):
                        path_folders = [p + '/' + p2 for p in pre for p2 in post]
                    path_folders = [[self.base_path +self.high_level_folders[0]+folder_name,self.base_path +self.high_level_folders[1]+folder_name,self.base_path +self.high_level_folders[2]+folder_name] for folder_name in path_folders]
                    print(path_folders)
                    [self.backend.draw_radar(fold, [pre[i],post[0]]) for i,fold in enumerate(path_folders)] 
                    self.plot_to_png(pre)

                    self.backend.clear_axes()
                    self.backend.legend=[]
        elif event == 'Go Advanced':
            key = []

    def close(self):
        self.window.close()

    def plot_to_png(self,fig_name=''):
        self.fig.canvas.draw()
        print(f'savimg to {fig_name}_{self.count}.png')
        plt.savefig(f'{fig_name}_{self.count}.png')
        self.count +=1
        return True

if __name__ == '__main__':
    thing = FigureGui()

    while True:
        thing.tick()
    thing.close()