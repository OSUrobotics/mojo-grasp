from mojograsp.simcore.reward import Reward
from mojograsp.simobjects.object_base import ObjectBase
import numpy as np
import pybullet as p
from mojograsp.simobjects.two_finger_gripper import TwoFingerGripper


class ExpertReward(Reward):
    def __init__(self):
        self.current_reward = {}

    def set_reward(self, goal_position: list, cube: ObjectBase, hand: TwoFingerGripper, end_reward):
        current_cube_pose = cube.get_curr_pose()
        # Finds distance between current cube position and goal position
        distance = np.sqrt((goal_position[0] - current_cube_pose[0][0])**2 +
                           (goal_position[1] - current_cube_pose[0][1])**2)
        
        f1_dist = p.getClosestPoints(cube.id, hand.id, 10, -1, 1, -1)
        f2_dist = p.getClosestPoints(cube.id, hand.id, 10, -1, 3, -1)
        # print(f2_dist)
        try:
            self.current_reward["distance_to_goal"] = distance
            self.current_reward["goal_position"] = goal_position
            self.current_reward["f1_dist"] = max(f1_dist[0][8], 0)
            self.current_reward["f2_dist"] = max(f2_dist[0][8],0)
            self.current_reward["end_penalty"] = end_reward
        except:
            print(f2_dist)
            

    def get_reward(self) -> dict:
        # print('reward', self.current_reward['distance_to_goal'])
        return self.current_reward.copy()



class TranslateReward(Reward):
    def __init__(self):
        self.current_reward = {}

    def set_reward(self, goal_position: list, cube: ObjectBase, hand: TwoFingerGripper, prev_position: list):
        current_cube_pose = cube.get_curr_pose()
        # Finds distance between current cube position and goal position
        f1_dist = p.getClosestPoints(cube.id, hand.id, 1, -1, 1, -1)
        f2_dist = p.getClosestPoints(cube.id, hand.id, 1, -1, 3, -1)
        

        distance = np.sqrt((goal_position[0] - current_cube_pose[0][0])**2 +
                           (goal_position[1] - current_cube_pose[0][1])**2) * 100
        distance_v = [100*(goal_position[0] - current_cube_pose[0][0]),
                           100*(goal_position[1] - current_cube_pose[0][1])]
        
        self.current_reward["distance_to_goal"] = distance
        self.current_reward["v_to_goal"] = distance_v
        self.current_reward["goal_position"] = goal_position
        self.current_reward["f1_dist"] = f1_dist[0][8]
        self.current_reward["f2_dist"] = f2_dist[0][8]
        self.current_reward["deltas"] = [100*(current_cube_pose[0][0]-prev_position[0][0]), 100*(current_cube_pose[0][1]-prev_position[0][1])]

    def get_reward(self) -> dict:
        # print('reward', self.current_reward['distance_to_goal'])
        return self.current_reward.copy()