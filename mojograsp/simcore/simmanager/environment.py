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
        phase.execute_action(phase.curr_action)
        self.step_sim(phase.curr_action)

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

    def get_obj_curr_pose(self, object_or_hand):
        return object_or_hand.get_curr_pose(object_or_hand.id)

    def get_obj_dimensions(self):
        return self.objects.get_dimensions()

    def get_curr_link_pos(self, link_id):
        """

        :param link_id: should be a joint index
        :return:
        """
        return self.hand.get_link_pose(link_id)
