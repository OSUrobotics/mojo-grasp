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
        self.sim_sleep = sleep
        self.sim_step = steps
        self.hand = hand
        self.objects = objects

    def reset(self):
        """
        Reset environment at the start of every new episode
        :return:
        """
        for i in self.hand.joint_dict.values():
            p.resetJointState(self.hand.id, i, 0)
        self.objects.set_curr_pose([0.00, 0.17, 0.0], self.objects.start_pos[self.objects.id][1])

    def step(self, phase):
        # With Action Profile
        phase.curr_action_profile = phase.Action.set_action_units(phase.curr_action)
        for i, j in zip(range(self.sim_step), phase.curr_action_profile):
            phase.execute_action(j)
            self.step_sim()

        # Without Action Profile:
        # for i in range(self.sim_step):
        #     phase.execute_action(phase.curr_action)
        #     self.step_sim()

        if phase.state is not None:
            observation = phase.state.update()
        else:
            observation = None
        # print("State:", observation)

        if phase.reward is not None:
            reward = phase.reward.get_reward()
        else:
            reward = None

        return observation, reward, None, None

    def step_sim(self):
        p.stepSimulation()
        time.sleep(self.sim_sleep)

    def get_hand_curr_joint_angles(self, keys=None):
        return self.hand.get_joint_angles(keys)

    def get_obj_curr_pose(self, object_or_hand):
        return object_or_hand.get_curr_pose(object_or_hand.id)

    def get_obj_dimensions(self):
        return self.objects.get_dimensions()

    def get_curr_link_pos(self, link_id):
        return self.hand.get_link_pose(link_id)

    def get_num_joints(self):
        return len(self.hand.joint_dict.values())

    def get_joint_nums_as_list(self):
        return list(self.hand.joint_dict.values())

    def get_contact_info(self, joint_index):
        return p.getContactPoints(self.objects.id, self.hand.id, linkIndexB=joint_index)
