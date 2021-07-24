#!/usr/bin/env python3
import time
import pybullet as p
from mojograsp.simcore.simmanager.environment_base import EnvironmentBase


class Environment(EnvironmentBase):
    """
    This class is used to reset the simulator's environment, execute actions, and step the simulator ahead
    """

    def __init__(self, action_class=None, sleep=1. / 240., steps=1, hand=None, objects=None):
        super().__init__(action_class)
        self.sleep = sleep
        self.steps = steps
        self.hand = hand
        self.objects = objects

    def reset(self):
        pass

    def step(self, phase):
        phase.execute_action(phase.action)
        self.step_sim(phase.action)

        if phase.state is not None:
            phase.state.update()
            observation = phase.state.get_obs()
        else:
            observation = None
        print("State:", observation)
        return observation, None, None, None

    def step_sim(self, action_profile):
        for i in range(self.steps):
            p.stepSimulation()
            time.sleep(self.sleep)

    def get_hand_curr_joint_angles(self, keys=None):
        return self.hand.get_joint_angles(keys)

    def get_obj_curr_pose(self, key):
        return self.objects[key].get_curr_pose(self.objects[key].id)
