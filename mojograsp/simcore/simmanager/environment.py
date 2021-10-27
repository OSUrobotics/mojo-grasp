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
        self.curr_timestep = 0
        self.curr_simstep = 0

    def reset(self):
        """
        Reset environment at the start of every new episode
        :return:
        """
        for i in self.hand.joint_dict.values():
            p.resetJointState(self.hand.id, i, 0)
        self.objects.set_curr_pose([0.00, 0.17, 0.0], self.objects.start_pos[self.objects.id][1])
        self.curr_timestep = 0
        self.curr_simstep = 0

    def step(self, phase):
        # With Action Profile
        self.curr_simstep = 0
        phase.curr_action_profile = phase.Action.set_action_units(phase.curr_action)
        for i, j in zip(range(self.sim_step), phase.curr_action_profile):
            self.curr_simstep += 1
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
            # print("TOTAL REWARD: {}".format(reward))
        else:
            reward = None

        return observation, reward, None, None

    def step_sim(self):
        p.stepSimulation()
        time.sleep(self.sim_sleep)

    def get_hand_curr_joint_angles(self, keys=None):
        return self.hand.get_joint_angles(keys)

    def get_hand_curr_pose(self):
        return self.hand.get_curr_pose()

    def get_obj_curr_pose(self):
        return self.objects.get_curr_pose()

    def get_obj_curr_pose_in_start_pose(self):
        curr_pos, curr_orn = self.get_obj_curr_pose()
        return self.objects.get_pose_in_start_pose(curr_pos, curr_orn)

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

    def set_obj_target_pose(self, direction):
        if direction == 'a':
            start_pose = self.objects.start_pos[self.objects.id]
            target_pos = (start_pose[0][0], start_pose[0][1] + 0.035, start_pose[0][2])
            target_orn = start_pose[1]
            angle_to_horizontal = -90
        else:
            print("Wrong Direction!")
            raise KeyError
        self.objects.target_pose = target_pos, target_orn
        self.objects.angle_to_horizontal = angle_to_horizontal

    def get_obj_target_pose(self):
        return self.objects.target_pose


