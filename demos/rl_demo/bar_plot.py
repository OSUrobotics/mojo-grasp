from mojograsp.simcore.data_gui_backend import *
import matplotlib.pyplot as plt

other_path = '/home/nigel/mojo-grasp/demos/rl_demo/data/Rot_90/Dynamic_90'
keys = ['square', 'square25','circle','circle25','triangle','triangle25','square_circle','pentagon','trapazoid']
folders = ['/' + k + '_A' for k in keys]

backend = PlotBackend()
all_data = []

for folder in folders:
    raw_values_m = backend.end_distance_return(other_path + folder)
    raw_values_cm = [val * 100 for val in raw_values_m] 
    all_data.append(raw_values_cm)
    backend.reset()

backend.clear_axes()
backend.ax.set_aspect('auto', adjustable='box')

plt.figure(figsize=(12, 6))
plt.boxplot(all_data, labels=keys, showfliers=False)
plt.xticks(rotation=45)
plt.title("Box Plot of End Distances for Dynamic Shapes")
plt.ylabel("End Distance")
plt.tight_layout()
plt.show()
