import numpy as np
import json
import matplotlib.pyplot as plt
import SiMBols as sb

class Evaluator():
    def __init__(self, data_folder: str):
        self.folder = data_folder
        # print('gone')

    def evaluate_single(self, episode_num: int):
        with open(self.folder + '/episode_' + str(episode_num)+'.json') as file:
            data_dict = json.load(file)
        data = data_dict['timestep_list']
        trajectory_points = [f['state']['obj_2']['pose'][0] for f in data]
        goal_pose = data[0]['reward']['goal_position']
        goal_pose[2] = 0.05
        start_pose = trajectory_points[0]

        full_trajectory = np.array(trajectory_points)
        real_trajectory = sb.Trajectory(full_trajectory)
        ideal_trajectory = np.linspace(start_pose,goal_pose,num=len(trajectory_points))
        ideal_trajectory = sb.Trajectory(ideal_trajectory)
        tester = sb.Comparer(real_trajectory,ideal_trajectory)
        tester.dfd()
        print(tester.DFD)
        rewards = [f['reward']['distance_to_goal'] for f in data]
        print('closest point to the goal', min(rewards))


    def evaluate_group(self, num_eps: list):
        for num in num_eps:
            with open(self.folder + 'episode_' + str(num)+'.json') as file:
                data_dict = json.load(file)
        