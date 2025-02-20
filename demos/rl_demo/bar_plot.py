from mojograsp.simcore.data_gui_backend import *

import matplotlib.pyplot as plt

base_path = '/home/mothra/mojo-grasp/demos/rl_demo/data/NTestLayer/Dynamic'
folders = [ base_path+'/square_A/',
            base_path+'/square25_A/',
            base_path+'/circle_A/',
            base_path+'/circle25_A/',
            base_path+'/triangle_A/',
            base_path+'/triangle25_A/',
            base_path+'/square_circle_A/',
            base_path+'/pentagon_A/',
            base_path+'/trapazoid_A/']

keys = ['square', 'square25','circle','circle25','triangle','triangle25','square_circle','pentagon','trapazoid']
backend = PlotBackend()
means = []
stds = []
for folder, k in zip(folders, keys):
    data = backend.draw_scatter_end_magic(folder)
    means.append(data[0])
    stds.append(data[1])
    backend.reset()
backend.clear_axes()
backend.ax.set_aspect('auto',adjustable='box')

plt.bar(keys,means)
plt.show()