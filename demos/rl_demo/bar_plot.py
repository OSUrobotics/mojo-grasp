from mojograsp.simcore.data_gui_backend import *

import matplotlib.pyplot as plt

base_path = '/home/mothra/mojo-grasp/demos/rl_demo/data/ReLu_Static'
other_path = '/home/mothra/mojo-grasp/demos/rl_demo/data/Static_1'
otherotherpath =  '/home/mothra/mojo-grasp/demos/rl_demo/data/Static_3'
folders = [ '/square_A/',
            '/square25_A/',
            '/circle_A/',
            '/circle25_A/',
            '/triangle_A/',
            '/triangle25_A/',
            '/square_circle_A/',
            '/pentagon_A/',
            '/trapazoid_A/']


keys = ['square', 'square25','circle','circle25',
        'triangle','triangle25','square_circle',
        'pentagon','trapazoid']

# keys = ['square','square15', 'square2', 'square3','circle', 'circle15', 'circle2','circle3','triangle', 'triangle15','triangle2','triangle3','teardrop',
            #  'teardrop15', 'teardrop2','teardrop3']

folders = ['/'+f+'_A' for f in keys]
backend = PlotBackend()
means = []
stds = []
means2 = []
m3 = []
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
    data = backend.draw_scatter_end_magic(otherotherpath + folder)
    m3.append(data[0])
    backend.reset()

backend.clear_axes()
backend.ax.set_aspect('auto',adjustable='box')

plt.bar(range(len(keys)),np.array(means)*1000,width=0.5)
plt.bar(np.array(range(len(keys)))+0.25,np.array(means2)*1000,width=0.5)
plt.bar(np.array(range(len(keys)))+0.5,np.array(m3)*1000,width=0.5)
# plt.bar(keys, means2, width=0.5)
plt.legend(['Static Trained on Square, Circle, Triangle', 'Bugged Static Trained on Square', 'Fixed Static'])
plt.ylabel('Average Ending Distanc (mm)')
plt.xlabel('Object')
plt.show()