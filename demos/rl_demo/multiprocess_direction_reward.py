from mojograsp.simcore.reward import RewardDefault
from mojograsp.simobjects.object_base import ObjectBase
import numpy as np
from mojograsp.simobjects.two_finger_gripper import TwoFingerGripper
from copy import deepcopy
from scipy.spatial.transform import Rotation as R

class MultiprocessDirectionReward(RewardDefault):
    def __init__(self, pybulletInstance=None):
        self.current_reward = {}
        self.prev_pos = []
        self.start_pos = []
        self.start_finger = []
        self.start_dist = None
        self.p = pybulletInstance
        self.timestep_num = 0

    def set_reward(self, goal_info: dict, cube: ObjectBase, hand: TwoFingerGripper, end_reward):
        #TODO add the finger contact goal distance stuff
        current_cube_pose = cube.get_curr_pose()
        # Finds distance between current cube position and goal position
        # print(goal_info)
        moved_path = -self.start_pos[0:2] +np.array(current_cube_pose[0][0:2])

        goal_direction = goal_info['goal_direction']

        dist_along_vec = np.dot(moved_path, goal_direction)

        dist_along_perp = abs(np.dot(moved_path, [-goal_direction[0],goal_direction[1]]))
        f1_dist = self.p.getClosestPoints(cube.id, hand.id, 10, -1, 1, -1)
        f2_dist = self.p.getClosestPoints(cube.id, hand.id, 10, -1, 4, -1)
        self.current_reward["f1_dist"] = max(f1_dist[0][8], 0)
        self.current_reward["f2_dist"] = max(f2_dist[0][8], 0)
        self.current_reward['dist_reward'] = dist_along_vec - 0.5*dist_along_perp
        self.current_reward['starting_orientation'] = self.start_orientation
        self.current_reward['dist_from_start'] = np.linalg.norm(moved_path)
        self.current_reward['object_orientation'] =  R.from_quat(current_cube_pose[1]).as_euler('xyz')

    def get_reward(self) -> dict:
        # print('reward', self.current_reward['distance_to_goal'])
        # print('getting the reward')
        return self.current_reward.copy()

    def setup_reward(self, start_state):
        self.prev_pos = start_state['obj_2']['pose']
        self.start_pos = np.array(start_state['obj_2']['pose'][0])
        self.start_orientation = R.from_quat(start_state['obj_2']['pose'][1]).as_euler('xyz')
        self.start_dist = None
        self.timestep_num = 0
        self.start_finger = [start_state['f1_pos'][0],start_state['f1_pos'][1],start_state['f2_pos'][0],start_state['f2_pos'][1]]

    def update_start(self, goal_info: dict, cube):
        pass
