#!/usr/bin/env python3
import time
import pybullet as p
from mojograsp.simcore.simmanager.environment_base import EnvironmentBase
import random


class Environment(EnvironmentBase):
    """
    This class is used to reset the simulator's environment, execute actions, and step the simulator ahead
    """
    """
    Environment reset will set the environment variable direction by randomly selecting between directions 
    [g, c, b, h, a], sub [1, 2, 3], trial [1, 2, 3, 4, 5]
    Phase 3 -> Expert/ RL Controller will get direction from environment variable
    """
    def __init__(self, action_class=None, sleep=1. / 240., steps=1, hand=None, objects=None, directions=None,
                 subjects=None, trials=None, trial_types=None):
        super().__init__(action_class)
        self.sim_sleep = sleep
        self.sim_step = steps
        self.hand = hand
        self.objects = objects
        self.curr_timestep = 0
        self.curr_simstep = 0
        self.directions = directions
        self.subjects = subjects
        self.trials = trials
        self.trial_types = trial_types
        self.curr_dir, self.curr_sub, self.curr_trial, self.curr_trial_type = None, None, None, None

        # TODO Make these accessible to user? or put else where
        # Dynamics
        roll_fric = 0.01
        # object
        p.changeDynamics(self.objects.id, -1, mass=0.04, rollingFriction=roll_fric)
        # distal
        p.changeDynamics(self.hand.id, 1, mass=0.03, rollingFriction=roll_fric)
        p.changeDynamics(self.hand.id, 3, mass=0.03, rollingFriction=roll_fric)
        # proximal
        p.changeDynamics(self.hand.id, 0, mass=0.02, rollingFriction=roll_fric)
        p.changeDynamics(self.hand.id, 2, mass=0.02, rollingFriction=roll_fric)

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
        self.curr_dir = random.choice(self.directions)
        self.curr_sub = random.choice(self.subjects)
        self.curr_trial = random.choice(self.trials)
        self.curr_trial_type = random.choice(self.trial_types)
        # print(self.curr_trial_type)
        self.set_obj_target_pose(self.curr_dir)


    def step(self, phase):
        self.curr_simstep = 0

        # With Action Profile
        phase.curr_action_profile = phase.Action.set_action_units(phase.curr_action)
        # print("\n", "\n", phase.curr_action, "\n", phase.curr_action_profile, "\n", "\n")
        for i, j in zip(range(self.sim_step), phase.curr_action_profile):
            self.curr_simstep += 1
            phase.execute_action(j)
            self.step_sim()

        # # Without Action Profile:
        # for i in range(self.sim_step):
        #     self.curr_simstep += 1
        #     phase.execute_action(phase.curr_action)
        #     self.step_sim()

        # print("Name is?: {} State is?: {}".format(phase.name, phase.state))
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
        return p.getClosestPoints(self.objects.id, self.hand.id, distance=10, linkIndexB=joint_index)

    # TODO: This needs to go
    def set_obj_target_pose(self, direction):
        start_pose = self.objects.start_pos[self.objects.id]
        target_orn = start_pose[1]
        if direction == 'a':
            change_in_x, change_in_y = 0, 0.035
        elif direction == 'b':
            change_in_x, change_in_y = 0.0247, 0.0247
        elif direction == 'c':
            change_in_x, change_in_y = 0.035, 0
        elif direction == 'd':
            change_in_x, change_in_y = 0.0247, -0.0247
        elif direction == 'e':
            change_in_x, change_in_y = 0, -0.035
        elif direction == 'f':
            change_in_x, change_in_y = -0.0247, -0.0247
        elif direction == 'g':
            change_in_x, change_in_y = -0.035, 0
        elif direction == 'h':
            change_in_x, change_in_y = -0.0247, 0.0247
        elif direction == 'a_b':
            change_in_x, change_in_y = 0.0134, 0.0323
        elif direction == 'b_c':
            change_in_x, change_in_y = 0.0323, 0.0134
        elif direction == 'c_d':
            change_in_x, change_in_y = 0.0323, -0.0134
        elif direction == 'd_e':
            change_in_x, change_in_y = 0.0123, -0.0323
        elif direction == 'e_f':
            change_in_x, change_in_y = -0.0123, -0.0323
        elif direction == 'f_g':
            change_in_x, change_in_y = -0.0323, -0.0134
        elif direction == 'g_h':
            change_in_x, change_in_y = -0.0323, 0.0134
        elif direction == 'h_a':
            change_in_x, change_in_y = -0.0134, 0.0323
        else:
            print("Wrong Direction!")
            raise KeyError
        target_pos = (start_pose[0][0] + change_in_x, start_pose[0][1] + change_in_y, start_pose[0][2])
        self.objects.target_pose = target_pos, target_orn
        self.objects.angle_to_horizontal = self.get_dir_angle()

    def get_obj_target_pose(self):
        return self.objects.target_pose

    def get_dir_angle(self):
        if self.curr_dir == 'a':
            angle = -1.5708
        elif self.curr_dir == 'b':
            angle = -0.7854
        elif self.curr_dir == 'c':
            angle = 0.0
        elif self.curr_dir == 'd':
            angle = 0.7854
        elif self.curr_dir == 'e':
            angle = 1.5708
        elif self.curr_dir == 'f':
            angle = 2.3562
        elif self.curr_dir == 'g':
            angle = 3.14
        elif self.curr_dir == 'h':
            angle = -2.3562
        elif self.curr_dir == 'a_b':
            angle = -1.1781
        elif self.curr_dir == 'b_c':
            angle = -0.3926
        elif self.curr_dir == 'c_d':
            angle = 0.3926
        elif self.curr_dir == 'd_e':
            angle = 1.1781
        elif self.curr_dir == 'e_f':
            angle = 1.9634
        elif self.curr_dir == 'f_g':
            angle = 2.7489
        elif self.curr_dir == 'g_h':
            angle = -2.7489
        elif self.curr_dir == 'h_a':
            angle = -1.9634
        else:
            print("Wrong Direction!")
            raise KeyError
        return [angle]


