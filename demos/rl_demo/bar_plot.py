from mojograsp.simcore.data_gui_backend import *

import matplotlib.pyplot as plt

base_path = '/home/mothra/mojo-grasp/demos/rl_demo/data/ReLu_Static'
other_path = '/home/mothra/mojo-grasp/demos/rl_demo/data/Dynamic_2'
folders = [ '/square_A/',
            '/square25_A/',
            '/circle_A/',
            '/circle25_A/',
            '/triangle_A/',
            '/triangle25_A/',
            '/square_circle_A/',
            '/pentagon_A/',
            '/trapazoid_A/']

keys = ['square', 'square25','circle','circle25','triangle','triangle25','square_circle','pentagon','trapazoid']
backend = PlotBackend()
means = []
stds = []
means2 = []
std2 = []
for folder, k in zip(folders, keys):
    data = backend.draw_scatter_end_magic(base_path + folder)
    means.append(data[0])
    stds.append(data[1])
    backend.reset()
    data = backend.draw_scatter_end_magic(other_path + folder)
    means2.append(data[0])
    std2.append(data[1])
    backend.reset()

backend.clear_axes()
backend.ax.set_aspect('auto',adjustable='box')

plt.bar(range(len(keys)),means,width=0.5)
plt.bar(np.array(range(len(keys)))+0.2,means2,width=0.5)
plt.legend(['ReLu Static', 'New Dynamic'])
plt.show()