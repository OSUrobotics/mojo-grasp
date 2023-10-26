from mojograsp.simcore.reward import RewardDefault
from mojograsp.simobjects.object_base import ObjectBase
import numpy as np
from mojograsp.simobjects.two_finger_gripper import TwoFingerGripper
from copy import deepcopy

class MultiprocessReward(RewardDefault):
    def __init__(self, pybullet_client):
        self.current_reward = {}
        self.prev_pos = []
        self.p = pybullet_client


    def set_reward(self, goal_position: list, cube: ObjectBase, hand: TwoFingerGripper, end_reward):
        current_cube_pose = cube.get_curr_pose()
        # Finds distance between current cube position and goal position
        distance = np.sqrt((goal_position[0] - current_cube_pose[0][0])**2 +
                           (goal_position[1] - current_cube_pose[0][1])**2)
        
        old_distance = np.sqrt((goal_position[0] - self.prev_pos[0][0])**2 +
                               (goal_position[1] - self.prev_pos[0][1])**2)
        
        # print('goal position', goal_position)
        # print('current cube pose', current_cube_pose)
        f1_dist = self.p.getClosestPoints(cube.id, hand.id, 10, -1, 1, -1)
        f2_dist = self.p.getClosestPoints(cube.id, hand.id, 10, -1, 4, -1)
        try:
            self.current_reward["distance_to_goal"] = distance
            self.current_reward["goal_position"] = goal_position
            self.current_reward["f1_dist"] = max(f1_dist[0][8], 0)
            self.current_reward["f2_dist"] = max(f2_dist[0][8], 0)
            self.current_reward["end_penalty"] = end_reward
            self.current_reward["slope_to_goal"] = old_distance - distance
            self.prev_pos = deepcopy(current_cube_pose)
            # print(f'old distance {old_distance}, current_distance {distance}')
        except:
            self.current_reward["distance_to_goal"] = distance
            self.current_reward["goal_position"] = goal_position
            self.current_reward["f1_dist"] = 10
            self.current_reward["f2_dist"] = 10
            self.current_reward["end_penalty"] = end_reward
            self.current_reward["slope_to_goal"] = old_distance - distance
            self.prev_pos = deepcopy(current_cube_pose)
            # print('we are 10 meters away or more. f1 and f2 dists set to 10 since we are already well below minimum possible reward')
            

    def get_reward(self) -> dict:
        # print('reward', self.current_reward['distance_to_goal'])
        return self.current_reward.copy()

    def setup_reward(self, start_pos):
        self.prev_pos = start_pos