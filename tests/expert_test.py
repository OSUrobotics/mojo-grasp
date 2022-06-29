import json
import pathlib
import matplotlib.pyplot as plt
import numpy as np

top_path = pathlib.Path().resolve()
data_path = top_path.parent.joinpath('demos/expert_demo/data')
file = str(data_path.joinpath('episode_all.json'))
with open(file) as json_file:
    expert_data = json.load(json_file)

obj_pos = {}
obj_orient = {}
goal_dists = {}
goal_poses = {}
max_and_min = [[100,-100],[100,-100]]
for episode in expert_data['episode_list']:
    obj_pos[episode['number']] = []
    obj_orient[episode['number']] = []
    goal_dists[episode['number']] = []
    goal_poses[episode['number']] = episode['timestep_list'][0]['reward']['goal_position']
    if episode['timestep_list'][0]['reward']['goal_position'][0] > max_and_min[0][1]:
        max_and_min[0][1] = episode['timestep_list'][0]['reward']['goal_position'][0]
    elif episode['timestep_list'][0]['reward']['goal_position'][0] < max_and_min[0][0]:
        max_and_min[0][0] = episode['timestep_list'][0]['reward']['goal_position'][0]
    if episode['timestep_list'][0]['reward']['goal_position'][1] > max_and_min[1][1]:
        max_and_min[1][1] = episode['timestep_list'][0]['reward']['goal_position'][1]
    elif episode['timestep_list'][0]['reward']['goal_position'][1] < max_and_min[1][0]:
        max_and_min[1][0] = episode['timestep_list'][0]['reward']['goal_position'][1]
    for timestep in episode['timestep_list']:
        obj_pos[episode['number']].append(timestep['state']['obj_2']['pose'][0])
        obj_orient[episode['number']].append(timestep['state']['obj_2']['pose'][1])
        goal_dists[episode['number']].append(timestep['reward']['distance_to_goal'])

    obj_orient[episode['number']] = np.array(obj_orient[episode['number']])
    obj_pos[episode['number']] = np.array(obj_pos[episode['number']])

goal_thing = []
episode_performance = [100]
for key, distance in goal_dists.items():
    episode_performance.append(np.min(distance))

performance_order = np.argsort(episode_performance)

n_plots = 5
legend = []
for i in range(n_plots):
    j = -i - 2
    # j = i
    goal_thing = [[0.0, goal_poses[performance_order[j]][0]], [0.16, goal_poses[performance_order[j]][1]]]
    plt.plot(obj_pos[performance_order[j]][:, 0], obj_pos[performance_order[j]][:, 1])
    plt.plot(goal_thing[0], goal_thing[1])
    legend.append('Actual Path '+ str(performance_order[j]))
    legend.append('Ideal Path '+ str(performance_order[j]))
plt.legend(legend)

plt.xlim(max_and_min[0])
plt.ylim(max_and_min[1])
plt.xlabel('X position (m)')
plt.ylabel('Y position (m)')
plt.show()
#
#
# plt.plot(range(len(goal_dists[6])),goal_dists[6])
# plt.show()

import glob
from PIL import Image


def make_gif(frame_folder, name='temporary_name'):
    frames = [Image.open(image) for image in glob.glob(f"{frame_folder}/*.png")]
    frame_one = frames[0]
    frame_one.save(name+".gif", format="GIF", append_images=frames,
                   save_all=True, duration=130, loop=0)

make_gif(data_path,'2400_speed')