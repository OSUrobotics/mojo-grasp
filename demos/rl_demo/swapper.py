import pickle as pkl

import numpy as np

# swap and negate


def mirror_action(filename):
    with open(filename,'rb') as file:
        episode_data = pkl.load(file)

    actions = np.array([a['action']['actor_output'] for a in episode_data])
    new_actions = np.array([-actions[:][-1],-actions[:][-2],-actions[:][-3],-actions[:][-4]])
    return new_actions