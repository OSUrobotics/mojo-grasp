from mojograsp.simcore.reward import Reward
from mojograsp.simobjects.object_base import ObjectBase
import numpy as np


class ExpertReward(Reward):
    def __init__(self):
        self.current_reward = {}

    def set_reward(self, goal_position: list, cube: ObjectBase):
        current_cube_pose = cube.get_curr_pose()
        # Finds distance between current cube position and goal position
        distance = np.sqrt((goal_position[0] - current_cube_pose[0][0])**2 +
                           (goal_position[1] - current_cube_pose[0][1])**2)

        self.current_reward["distance_to_goal"] = distance
        self.current_reward["goal_position"] = goal_position

    def get_reward(self) -> dict:
        return self.current_reward.copy()
