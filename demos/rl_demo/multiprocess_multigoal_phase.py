from mojograsp.simcore.phase import Phase
from mojograsp.simobjects.two_finger_gripper import TwoFingerGripper
from mojograsp.simobjects.object_base import ObjectBase
from mojograsp.simcore.state import State
from mojograsp.simcore.reward import Reward
from mojograsp.simcore.action import Action
from demos.rl_demo import multiprocess_control
from mojograsp.simcore.replay_buffer import ReplayBufferDefault
from numpy.random import shuffle
from math import isclose
from PIL import Image
from demos.rl_demo.multiprocess_manipulation_phase import MultiprocessManipulation

class MultigoalManipulation(MultiprocessManipulation):
    def __init__(self, hand: TwoFingerGripper, cube: ObjectBase, state: State, action: Action, reward: Reward, env, args: dict = None, physicsClientId=None, hand_type=None):
        super().__init__(hand, cube, state, action, reward, env, None, args, physicsClientId, hand_type)

    def post_step(self):
        #check for goal achievements
        # print('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')
        self.state.check_goal()
        # Set the reward from the given action after the step
        self.reward.set_reward(self.state.get_goal(), self.cube, self.hand, self.controller.final_reward)

    def exit_condition(self, eval_exit=False) -> bool:
        goals = self.state.get_goal()
        # print(goals['timesteps_remaining'])
        if (goals['timesteps_remaining'] <= 0) | (sum(goals['goals_open'])==0):
            # print(goals['timesteps_remaining'])
            return True
        else:
            return False