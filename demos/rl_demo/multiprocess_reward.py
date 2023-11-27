from mojograsp.simcore.reward import RewardDefault
from mojograsp.simobjects.object_base import ObjectBase
import numpy as np
from mojograsp.simobjects.two_finger_gripper import TwoFingerGripper
from copy import deepcopy


class MultiprocessReward(RewardDefault):
    def __init__(self, pybulletInstance=None):
        self.current_reward = {}
        self.prev_pos = []
        self.start_pos = []
        self.start_dist = None
        self.p = pybulletInstance

    def set_reward(self, goal_position: list, cube: ObjectBase, hand: TwoFingerGripper, end_reward):
        current_cube_pose = cube.get_curr_pose()
        # Finds distance between current cube position and goal position
        distance = np.sqrt((goal_position[0] - current_cube_pose[0][0])**2 +
                           (goal_position[1] - current_cube_pose[0][1])**2)
        
        old_distance = np.sqrt((goal_position[0] - self.prev_pos[0][0])**2 +
                               (goal_position[1] - self.prev_pos[0][1])**2)
        
        f1_dist = self.p.getClosestPoints(cube.id, hand.id, 10, -1, 1, -1)
        f2_dist = self.p.getClosestPoints(cube.id, hand.id, 10, -1, 4, -1)
        
        velocity = cube.get_curr_velocity()
        
        start_pos_vec = np.array(goal_position[0:2])-self.start_pos[0:2]
        
        current_pos_vec = np.array(goal_position[0:2]) - np.array([current_cube_pose[0][0],current_cube_pose[0][1]])
        
        if self.start_dist is None:
            self.start_dist = distance.copy()
        
        self.current_reward['object_velocity'] = velocity[0]
        self.current_reward['start_dist'] = self.start_dist
        self.current_reward['plane_side'] = np.dot(start_pos_vec,current_pos_vec) <= 0
        # print('setting the reward')
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
        # print('getting the reward')
        return self.current_reward.copy()

    def setup_reward(self, start_pos):
        self.prev_pos = start_pos
        self.start_pos = np.array(start_pos[0])
        self.start_dist = None
        # print(self.start_pos)

    def update_start(self, goal_position, cube):
        current_cube_pose = cube.get_curr_pose()
        # Finds distance between current cube position and goal position
        distance = np.sqrt((goal_position[0] - current_cube_pose[0][0])**2 +
                           (goal_position[1] - current_cube_pose[0][1])**2)
        self.start_dist = distance
