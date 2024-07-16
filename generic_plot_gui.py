import PySimpleGUI as sg
import os
import numpy as np
import re
import pathlib
from mojograsp.simcore.data_gui_backend import PlotBackend
from PIL import ImageGrab
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pickle as pkl

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


def load_data(filepath):
    with open(filepath, 'rb') as file:
        data = pkl.load(file)
    if type(data) is list:
        data = {'timestep_list': data, 'number':0}
    return data

def build_tree(data, tree,upperkey):
    for key, value in data.items():
        if isinstance(value, dict):
            tree.Insert(upperkey, key, key, values=[])
            build_tree(value, tree, key)
        else:
            tree.Insert(upperkey, key, key, values=[value])

def main():
    toggle_btn_off = b'iVBORw0KGgoAAAANSUhEUgAAACgAAAAoCAYAAACM/rhtAAAABmJLR0QA/wD/AP+gvaeTAAAED0lEQVRYCe1WTWwbRRR+M/vnv9hO7BjHpElMKSlpqBp6gRNHxAFVcKM3qgohQSqoqhQ45YAILUUVDRxAor2VAweohMSBG5ciodJUSVqa/iikaePEP4nj2Ovdnd1l3qqJksZGXscVPaylt7Oe/d6bb9/svO8BeD8vA14GvAx4GXiiM0DqsXv3xBcJU5IO+RXpLQvs5yzTijBmhurh3cyLorBGBVokQG9qVe0HgwiXLowdy9aKsY3g8PA5xYiQEUrsk93JTtjd1x3siIZBkSWQudUK4nZO1w3QuOWXV+HuP/fL85klAJuMCUX7zPj4MW1zvC0Ej4yMp/w++K2rM9b70sHBYCjo34x9bPelsgp/XJksZ7KFuwZjr3732YcL64ttEDw6cq5bVuCvgy/sje7rT0sI8PtkSHSEIRIKgCQKOAUGM6G4VoGlwiqoVd2Za9Vl8u87bGJqpqBqZOj86eEHGNch+M7otwHJNq4NDexJD+59RiCEQG8qzslFgN8ibpvZNsBifgXmFvJg459tiOYmOElzYvr2bbmkD509e1ylGEZk1Y+Ssfan18n1p7vgqVh9cuiDxJPxKPT3dfGXcN4Tp3dsg/27hUQs0qMGpRMYjLz38dcxS7Dm3nztlUAb38p0d4JnLozPGrbFfBFm79c8hA3H2AxcXSvDz7/+XtZE1kMN23hjV7LTRnKBh9/cZnAj94mOCOD32gi2EUw4FIRUMm6LGhyiik86nO5NBdGRpxYH14bbjYfJteN/OKR7UiFZVg5T27QHYu0RBxoONV9W8KQ7QVp0iXdE8fANUGZa0QAvfhhXlkQcmjJZbt631oIBnwKmacYoEJvwiuFgWncWnXAtuVBBEAoVVXWCaQZzxmYuut68b631KmoVBEHMUUrJjQLXRAQVSxUcmrKVHfjWWjC3XOT1FW5QrWpc5IJdQhDKVzOigEqS5dKHMVplnNOqrmsXqUSkn+YzWaHE9RW1FeXL7SKZXBFUrXW6jIV6YTEvMAUu0W/G3kcxPXP5ylQZs4fa6marcWvvZfJu36kuHjlc/nMSuXz+/ejxgqPFpuQ/xVude9eu39Jxu27OLvBGoMjrUN04zrNMbgVmOBZ96iPdPZmYntH5Ls76KuxL9NyoLA/brav7n382emDfHqeooXyhQmARVhSnAwNNMx5bu3V1+habun5nWdXhwJZ2C5mirTesyUR738sv7g88UQ0rEkTDlp+1wwe8Pf0klegUenYlgyg7bby75jUTITs2rhCAXXQ2vwxz84vlB0tZ0wL4NEcLX/04OrrltG1s8aOrHhk51SaK0us+n/K2xexBxljcsm1n6x/Fuv1PCWGiKOaoQCY1Vb9gWPov50+fdEqd21ge3suAlwEvA14G/ucM/AuppqNllLGPKwAAAABJRU5ErkJggg=='
    toggle_btn_on = b'iVBORw0KGgoAAAANSUhEUgAAACgAAAAoCAYAAACM/rhtAAAABmJLR0QA/wD/AP+gvaeTAAAD+UlEQVRYCe1XzW8bVRCffbvrtbP+2NhOD7GzLm1VoZaPhvwDnKBUKlVyqAQ3/gAkDlWgPeVQEUCtEOIP4AaHSI0CqBWCQyXOdQuRaEFOk3g3IMWO46+tvZ+PeZs6apq4ipON1MNafrvreTPzfvub92bGAOEnZCBkIGQgZOClZoDrh25y5pdjruleEiX+A+rCaQo05bpuvJ/+IHJCSJtwpAHA/e269g8W5RbuzF6o7OVjF8D3Pr4tSSkyjcqfptPDMDKSleW4DKIggIAD5Yf+Oo4DNg6jbUBlvWLUNutAwZu1GnDjzrcXzGcX2AHw/emFUV6Sfk0pqcKpEydkKSo9q3tkz91uF5aWlo1Gs/mYc+i7tz4//19vsW2AU9O381TiioVCQcnlRsWeQhD3bJyH1/MiFLICyBHiuzQsD1arDvypW7DR9nzZmq47q2W95prm+I9fXfqXCX2AF2d+GhI98Y8xVX0lnxvl2UQQg0csb78ag3NjEeD8lXZ7pRTgftmCu4864OGzrq+5ZU0rCa3m+NzXlzvoAoB3+M+SyWQuaHBTEzKMq/3BMbgM+FuFCDBd9kK5XI5PJBKqLSev+POTV29lKB8rT0yMD0WjUSYLZLxzNgZvIHODOHuATP72Vwc6nQ4Uiw8MUeBU4nHS5HA6TYMEl02wPRcZBJuv+ya+UCZOIBaLwfCwQi1Mc4QXhA+PjWRkXyOgC1uIhW5Qd8yG2TK7kSweLcRGKKVnMNExWWBDTQsH9qVmtmzjiThQDs4Qz/OUSGTwcLwIQTLW58i+yOjpXDLqn1tgmDzXzRCk9eDenjo9yhvBmlizrB3V5dDrNTuY0A7opdndStqmaQLPC1WCGfShYRgHdLe32UrV3ntiH9LliuNrsToNlD4kruN8v75eafnSgC6Luo2+B3fGKskilj5muV6pNhk2Qqg5v7lZ51nBZhNBjGrbxfI1+La5t2JCzfD8RF1HTBGJXyDzs1MblONulEqPDVYXgwDIfNx91IUVbAbY837GMur+/k/XZ75UWmJ77ou5mfM1/0x7vP1ls9XQdF2z9uNsPzosXPNFA5m0/EX72TBSiqsWzN8z/GZB08pWq9VeEZ+0bjKb7RTD2i1P4u6r+bwypo5tZUumEcDAmuC3W8ezIqSGfE6g/sTd1W5p5bKjaWubrmWd29Fu9TD0GlYlmTx+8tTJoZeqYe2BZC1/JEU+wQR5TVEUPptJy3Fs+Vkzgf8lemqHumP1AnYoMZSwsVEz6o26i/G9Lgitb+ZmLu/YZtshfn5FZDPBCcJFQRQ+8ih9DctOFvdLIKHH6uUQnq9yhFu0bec7znZ+xpAGmuqef5/wd8hAyEDIQMjAETHwP7nQl2WnYk4yAAAAAElFTkSuQmCC'

    # Get the folder containing the episodes
    p1 = pathlib.Path(__file__).parent.resolve()
    folder = sg.popup_get_folder('Episode Folder to open',initial_folder=str(p1)+'/demos/rl_demo/data')
    if folder is None:
        sg.popup_cancel('Cancelling')
        return

    # get list of pkl files in folder
    episode_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith('.pkl')]
    filenames_only = [f for f in os.listdir(folder) if f.lower().endswith('.pkl')]
    
    filenums = [re.findall('\d+',f) for f in filenames_only]
    final_filenums = []
    for i in filenums:
        if len(i) > 0 :
            final_filenums.append(int(i[-1]))
        else:
            final_filenums.append(10000000000)
    if len(episode_files) == 0:
        sg.popup('No pkl episodes in folder, using json format.')
        episode_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith('.json')]
        filenames_only = [f for f in os.listdir(folder) if f.lower().endswith('.json')]


    screen_width, screen_height = sg.Window.get_screen_size()
    sorted_inds = np.argsort(final_filenums)
    final_filenums = np.array(final_filenums)
    temp = final_filenums[sorted_inds]
    episode_files = np.array(episode_files)
    filenames_only = np.array(filenames_only)

    episode_files = episode_files[sorted_inds].tolist()
    filenames_only = filenames_only[sorted_inds].tolist()
    folder_location = os.path.abspath(episode_files[0])
    overall_path = pathlib.Path(folder_location).parent.resolve()
    # define menu layout
    menu = [['File', ['Open Folder', 'Exit']], ['Help', ['About', ]]]

    scatter_plot_tab = [[sg.Button('Goal Wizard', size=(8,2)), sg.Button('Goal Spell', size=(8,2)), sg.Button('End Poses', size=(8, 2)), sg.Button('Ending Distances', size=(8,2)), sg.Button('Path Spell', size=(8,2))],
                        [sg.Button('Orientation Wizard', size=(8,2)), sg.Button('Orientation Scatter Spell', size=(8,2)), sg.Button('Orientation Multi', key='Orientation Multi',size=(8, 2)), sg.Button('Rotation Sliding Error', size=(8,2)), sg.Button('Orientation Spell',size=(8,2))],
                        [sg.Button('OR bucket', size=(8,2)), sg.Button('Orientation Single Region', size=(8, 2)), sg.Button('Success Scatter',size=(8,2)), sg.Button('Shenanigans',size=(8,2)), sg.Button('Contact Spell',size = (8,2))],
                        [sg.Button('Explored Region', size=(8,2)), sg.Button('Reward Comparison', size=(8,2)), sg.Button('Timestep Best',size=(8,2)), sg.Button('Finger Object Avg', size=(8,2)),sg.Button('Scatter Scaled',size=(8,2))],
                        [sg.Button('End Region', size=(8,2)), sg.Button('Max Percent', size=(8,2)),sg.Button('Timestep End',size=(8,2)), sg.Button('Finger Object Max', size=(8,2)), sg.Button('Success Rate', size=(8,2))]]

    plot_buttons = [[sg.Button('Object Path', size=(8, 2)), sg.Button('Finger Angles', size=(8, 2)), sg.Button('Finger Contact Distance',size=(8, 2)), sg.Button('Rewards', size=(8, 2)),sg.Button('BINGO',size=(8,2))],
                    [sg.Button('Fingertip Path', size=(8,2)), sg.Button('Actor Output', size=(8, 2)), sg.Button('Object Goal Distance',size=(8, 2)),sg.Button('Big Success',size=(8,2)),sg.Button('Load Dictionary',size=(8,2))],
                    [sg.Button('Obj Contacts', size=(8,2)), sg.Button('Aout Comparison', size=(8, 2)),sg.Button('Orientation', size=(8,2)), sg.Button('Multireward', size=(8,2)),sg.Button('Save Dictionary',size=(8,2))],
                    [sg.Button('Finger Goal Path',size=(8,2)),sg.Button('Sampled Poses', size=(8,2)),sg.Button('draw_scatter_max_end',size=(8,2)),sg.Button('Both Errors',size=(8,2)),sg.Button('draw_newshit',size=(8,2))],
                    [sg.Button('draw_fuckery'), sg.Button('draw_z'), sg.Button('draw_boxen'), sg.Button('b2')]]
    tree = sg.TreeData()
    build_tree({'':None},tree,'')
    json_column = [[sg.Tree(data=tree, headings=[], auto_size_columns=True, num_rows=20, col0_width=40,key='json_info')]]
    # define layout, show and read the window
    col = [[sg.Text(episode_files[0], size=(80, 3), key='-FILENAME-')],
           [sg.Canvas(size=(1280, 960), key='-CANVAS-')],
           [sg.TabGroup([[sg.Tab('Standard Plotting', plot_buttons)],
                        [sg.Tab('Scatter Plotting', scatter_plot_tab)]],key='-group1-', tab_location='top', selected_title_color='purple')], [sg.Input('Temp',key='save_name'),sg.B('Save Image', key='-SAVE-')],
               [sg.Text('File 1 of {}'.format(len(episode_files)), size=(15, 1), key='-FILENUM-')]]

    # scatter_col = [[sg.Text(episode_files[0], size=(80, 3), key='-FILENAME-')],
    #        [sg.Canvas(size=(1280, 960), key='-CANVAS-')],
    #        scatter_plot_tab[0], [sg.Input('Temp',key='save_name'),sg.B('Save Image', key='-SAVE-')],
    #            [sg.Text('File 1 of {}'.format(len(episode_files)), size=(15, 1), key='-FILENUM-')]]

    col_files = [[sg.Text(overall_path, key='-print-path')],
                 [sg.Button('Switch Train/Test'),sg.Button('Select New Folder')],
                 [sg.Listbox(values=filenames_only, size=(60, 30), key='-LISTBOX-', enable_events=True)],
                 [sg.Text('Select an episode.  Use scrollwheel or arrow keys on keyboard to scroll through files one by one.')],
                 [sg.Text('Timestep for Completion'), sg.Input(15,size=(5,1),key='tstep')],
                 [sg.Text('Reward Function'),sg.OptionMenu(values=('Sparse','Distance','Distance + Finger', 'Hinge Distance + Finger', 'Slope', 'Slope + Finger','SmartDistance + Finger','SmartDistance + SmartFinger','ScaledDistance + Finger','ScaledDistance+ScaledFinger', 'SFS','DFS','TripleScaled',"full", "full+finger","Rotation", "Rotation+Finger", "continuous_finger", "end_finger"), k='-rf',default_value='TripleScaled')],
                 [sg.Text('Colormap'),sg.Input('plasma_r',key='-cmap',size=(8, 1))],
                 [sg.Text('Num Averaged'),sg.Input(1200,key='moving_avg',size=(8,2)), sg.Text("Keep previous graph", key='-toggletext-'), sg.Button(image_data=toggle_btn_off, key='-TOGGLE-GRAPHIC-', button_color=(sg.theme_background_color(), sg.theme_background_color()), border_width=0, metadata=False)],
                 [sg.Text('Translational Success Threshold (mm)'),sg.Input(10,key='success_range', size=(8,1))],
                 [sg.Text('Rotational Success Threshold (deg)'), sg.Input(10, key='rot_success_range',size=(8,1))],
                 [sg.Text("Distance Scale"),  sg.Input(1,key='-distance_scale',size=(5, 1)), sg.Text('Contact Scale'), sg.Input(0.2,key='-contact_scale',size=(5, 1))],  
                 [sg.Text('Rotation Scale'),sg.Input(1,key='-rotation_scale',size=(5, 1)), sg.Text('Success Reward'), sg.Input(1,key='-success_reward',size=(5, 1))],
                 [sg.Text('Action Type'),sg.OptionMenu(values=('FTP','JA'), k='-atype',default_value='FTP')],
                 [sg.Text('Size'),sg.Input(30,key='markersize',size=(5,1)), sg.Text('Separation Distance'), sg.Input(0.2, key='sepdist',size=(5,1))]]

    layout = [[sg.Menu(menu)], [sg.Col(col_files),sg.Col(json_column), sg.Col(col)]]
    # layout = [[sg.Menu(menu)], [sg.TabGroup([[sg.Tab('Single Plotting', [[sg.Col(col_files), sg.Col(col)]]),
    #                         sg.Tab('Scatter Plotting', [[sg.Col(col_files), sg.Col(scatter_col)]])]], key='-group1-', tab_location='top', selected_title_color='purple')]]

    window = sg.Window('Analysis Window', layout, return_keyboard_events=True, use_default_focus=False, finalize=True)
    
    window.move(1000, 20)
    canvas = window['-CANVAS-'].TKCanvas

    temp = pathlib.Path(folder).parent.resolve()
    backend = PlotBackend()
    fig, _ = backend.get_figure()
    figure_canvas_agg = FigureCanvasTkAgg(fig, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    tholds = {'SUCCESS_THRESHOLD':0,
                       'DISTANCE_SCALING':0,
                       'CONTACT_SCALING':0,
                       'SUCCESS_REWARD':0}
    rf_key = ''
    # input('initialized backend')
    # loop reading the user input and displaying image, filename
    filenum, filename = 0, episode_files[0]

    episode_data = load_data(filename)
    class click_thingy:
        def __init__(self):
            self.x = None
            self.y = None

        def reset(self):
            self.x = None
            self.y = None

        def callback(self,event):
            self.x=event.xdata
            self.y=event.ydata
            print(self.x,self.y)
    clicks = click_thingy()
    figure_canvas_agg.callbacks.connect('button_press_event', clicks.callback)

    tree = sg.TreeData()
    build_tree(episode_data['timestep_list'][0],tree,'')
    window['json_info'].update(tree)

    # input('about to start loop')
    # TODO: add in functionality to check which type of file it is and send in either the epidoe data OR the string to the folder
    counter = 0
    while True:
        event, values = window.read()
        # ----------------- Menu choices -----------------
        if event == 'Open Folder':
            newfolder = sg.popup_get_folder('Episode Folder to open',initial_folder=str(p1)+'/demos/rl_demo/data')
            if newfolder is None:
                continue

            folder = newfolder
            episode_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith('.pkl')]
            filenames_only = [f for f in os.listdir(folder) if f.lower().endswith('.pkl')]
            backend.reset()
            print('we just called backend.reset')
            window['-LISTBOX-'].update(values=filenames_only)
            window.refresh()

            filenum = 0
        print(event, values)
        # update window with filename
        window['-FILENAME-'].update(filename)
        # update page display
        window['-FILENUM-'].update('File {} of {}'.format(filenum + 1, len(episode_files)))
        if (event == sg.WIN_CLOSED) | (event =='Exit'):
            break

    window.close()
if __name__ == '__main__':
    main()