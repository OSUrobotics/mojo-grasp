from data_gui_backend import PlotBackend
import pickle as pkl
import pathlib
import os
import matplotlib.pyplot as plt
from fpdf import FPDF


def load_data(filepath):
    with open(filepath, 'rb') as file:
        data = pkl.load(file)
    return data

def build_first_argument(folder_path, arg_dict):
    file_path = os.path.join(folder_path, arg_dict['folder'])
    episode_files = os.listdir(file_path)
    if 'episode_number' in arg_dict.keys():
        filename = [name for name in episode_files if (str(arg_dict['episode_number'])+'.pkl' in name)]
        print(filename)
        if len(filename) ==1:
            datafile = load_data(os.path.join(file_path, filename[0]))
            return datafile
        else:
            raise IndexError('No episode with episode number'+str(arg_dict['episode_number']))
    else:
        
        if 'episode_all.pkl' in episode_files:
            datafile = load_data(os.path.join(file_path, 'episode_all.pkl'))
            return datafile
        else:
            return file_path

plots_to_generate = [{'type':'net_reward','moving_average':20, 'folder':'Train'},
                     [{'type':'success_rate','moving_average':20, 'folder':'Train','success_threshold':10},{'type':'success_rate','moving_average':20, 'folder':'Train','success_threshold':5}],
                     [{'type':'ending_goal_dist','moving_average':20, 'folder':'Train'},{'type':'shortest_goal_dist','moving_average':20, 'folder':'Train'}]]
text_to_add = ['Net reward for this direction averaged across 20 episodes. Demonstrates the overall training.',
               'Success rate for this direction for both 5mm and 10 mm. Ideal should have full success at 5mm by end',
               'Ending and minimum goal distance. Main purpose is to verify that both are reducing consistantly']
method_list = [attribute for attribute in dir(PlotBackend) if callable(getattr(PlotBackend, attribute)) and attribute.startswith('draw') is True]
print(method_list)

valid_types = [a[5:] for a in method_list]
high_level = '/home/orochi/mojo/mojo-grasp/demos/rl_demo/data/wedge'

mid_levels = os.listdir(high_level)
folder_path = os.path.join(high_level,mid_levels[0])
plotter = PlotBackend(folder_path)
plotter.clear_plots=False
for direction in mid_levels:
    folder_path = os.path.join(high_level,direction)
    plotter.load_config(folder_path)
    for i,desired_plot in enumerate(plots_to_generate):
        if type(desired_plot) is dict:
            if desired_plot['type'] in valid_types:
                function_to_run = getattr(plotter,'draw_'+desired_plot['type'])
                if 'moving_average' in desired_plot.keys():
                    plotter.moving_avg = desired_plot['moving_average']
                arg1 = build_first_argument(folder_path,desired_plot)
                if 'success_threshold' in desired_plot.keys():
                    function_to_run(arg1, desired_plot['success_threshold']/1000)
                else:
                    function_to_run(arg1)
        else:
            for d_plot in desired_plot:
                if d_plot['type'] in valid_types:
                    arg1 = build_first_argument(folder_path,d_plot)
                    function_to_run = getattr(plotter,'draw_'+d_plot['type'])
                    if 'moving_average' in d_plot.keys():
                        plotter.moving_avg = d_plot['moving_average']
                    if 'success_threshold' in d_plot.keys():
                        function_to_run(arg1, d_plot['success_threshold']/1000)
                    else:
                        function_to_run(arg1)
        plt.savefig(os.path.join(folder_path,'Plots','fig_'+str(i)+'.png'))
        plotter.clear_axes()
        num_plots = plt.get_fignums()
        if len(num_plots) > 1:
            plt.close(num_plots[-1])
    pdf_thing = FPDF()
    pdf_thing.set_font('Times', '', 12)
    for i,text in enumerate(text_to_add):
        pdf_thing.add_page()
        pdf_thing.cell(40, 10, text)
        pdf_thing.image(os.path.join(folder_path,'Plots','fig_'+str(i)+'.png'),25,20,160)
        
    pdf_thing.output(os.path.join(folder_path,'Plots','AutoGenerated.pdf'), 'F')