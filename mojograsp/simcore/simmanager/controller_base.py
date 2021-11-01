#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 11:14:16 2021
@author: orochi
"""
import numpy as np
import os
import mojograsp
import pybullet as p
import pandas as pd
from math import radians


class ControllerBase:
    _sim = None

    def __init__(self, state_path=None):
        """

        :param state_path: This can be the path to a json file or a pointer to an instance of state
        """
        if '.json' in state_path:
            self.state = mojograsp.state_space.StateSpace(path=state_path)
        else:
            self.state = state_path

    def select_action(self):
        pass

    @staticmethod
    def create_instance(state_path, controller_type):
        """
        Create a Instance based on the contorller type
        :param state_path: is the json file path or instance of state space
        :param controller_type: string type, name of controller defining which controller instance to create
        :return: An instance based on the controller type
        """
        if controller_type == "open":
            return OpenController(state_path)
        elif controller_type == "close":
            return CloseController(state_path)
        elif controller_type == "move":
            return MoveController(state_path)
        elif controller_type == "rl":
            return DDPGfD(state_path)
        else:
            controller = controller_type
            try:
                controller.select_action
            except AttributeError:
                raise AttributeError('Invalid controller type. '
                                     'Valid controller types are: "open", "close" and "PID move"')
            return controller


class OpenController(ControllerBase):
    def __init__(self, state_path):
        super().__init__(state_path)

    def select_action(self):
        """
        This controller is defined to open the hand.
        Thus the action will always be a constant action
        of joint position to be reached
        :return: action: action to be taken in accordance with action space
        """
        action = [1.57, 0, -1.57, 0]
        return action


class CloseController(ControllerBase):
    def __init__(self, state_path):
        super().__init__(state_path)

    def select_action(self):
        """
        This controller is defined to close the hand.
        Thus the action will always be a constant action
        of joint position to be reached
        :return: action: action to be taken in accordance with action space
        """
        action = [0.78, -1.65, -0.78, 1.65]
        return action


class MoveController(ControllerBase):
    def __init__(self, state_path):
        super().__init__(state_path)
        self.dir = 'c'
        ControllerBase._sim.set_obj_target_pose(self.dir)
        self.filename = "/Users/asar/PycharmProjects/InHand-Manipulation/Human Study Data/" \
                        "asterisk_test_data_for_anjali/trial_paths/not_normalized/sub1_2v2_{}_n_1.csv".format(self.dir)
        ControllerBase._sim.set_obj_target_pose(self.dir)
        self.object_poses_expert = self.extract_data_from_file()
        self.iterator = 0
        self.data_len = len(self.object_poses_expert)
        self.data_over = False

    def extract_data_from_file(self):
        """
        Read in csv file  containing  information  of human studies as a panda dataframe.
        Convert it  to numpy arrays
        Format: Start pos of hand is origin
        x,y,rmag,f_x,f_y,f_rmag
        Note: c dir is +x; a dir is +y [from humanstudy data]
        :param filename: Name of file containing human data
        :return: numpy  array containing the information from the file
        """
        df = pd.read_csv(self.filename)
        df.drop(df.columns[df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
        # print("Head of file: \n", df.head())
        data = df.to_numpy()
        return data

    def get_next_line(self):
        next_pose = self.object_poses_expert[self.iterator]
        self.iterator += 1
        if self.iterator >= self.data_len:
            self.data_over = True
        return next_pose

    @staticmethod
    def _convert_data_to_pose(data, scale=0.001):
        """
        TODO: Change name to better represent what's happening here "convert_data_to_pose"?
        Get the next pose values from the file in the proper pose format
        :param data: Data line from file as a list [x, y, rotx,  f_x,f_y,f_rot_mag]
        :return:
        """
        pos = (data[4] * scale, data[5] * scale, 0.0)
        orn_eul = [0, 0, radians(data[6])]
        orn = p.getQuaternionFromEuler(orn_eul)
        return pos, orn

    def _get_origin_cube(self, data):
        """
        TODO: make this private?
        Gets the next pose of the  cube in world coordinates
        :param cube:
        :param data:
        :return:
        """
        next_pos, next_orn = self._convert_data_to_pose(data)
        T_origin_next_pose_cube = p.multiplyTransforms(self._sim.objects.start_pos[self._sim.objects.id][0],
                                                       self._sim.objects.start_pos[self._sim.objects.id][1], next_pos,
                                                       next_orn)
        return T_origin_next_pose_cube

    def get_contact_points(self, cube_id):
        """
        Get the contact points between object passed in (cube) and gripper
        If no  contact, returns None
        :param cube_id:
        :return: [contact_points_left, contact_points_right] left and right finger contacts
        """
        contact_points = []
        for i in range(0, len(self._sim.hand.end_effector_indices)):
            contact_points_info = p.getContactPoints(cube_id, self._sim.hand.id,
                                                     linkIndexB=self._sim.hand.end_effector_indices[i])
            try:
                contact_points.append(contact_points_info[0][6])
            except IndexError:
                contact_points.append(None)

        return contact_points

    def maintain_contact(self):
        # print("No Contact")
        target_pos = []
        go_to, _ = self._sim.objects.get_curr_pose()
        for j in range(0, len(self._sim.hand.end_effector_indices)):
            target_pos.append(go_to)
        return target_pos

    def get_origin_cp(self, i, cube, T_cube_origin, T_origin_nextpose_cube, curr_contact_points):
        """
        TODO: make this private?
        :return:
        """
        pos, curr_obj_orn = self._sim.objects.get_curr_pose()
        # curr_contact_points[i] = 0,0,0
        # print("Current Contacts: {}\nCurrent Orientation: {}".format(curr_contact_points[i], curr_obj_orn))
        T_cube_cp = p.multiplyTransforms(T_cube_origin[0], T_cube_origin[1], curr_contact_points[i], curr_obj_orn)
        T_origin_new_cp = p.multiplyTransforms(T_origin_nextpose_cube[0], T_origin_nextpose_cube[1],
                                               T_cube_cp[0], T_cube_cp[1])

        return T_origin_new_cp

    def get_origin_links(self, i, j, T_origin_newcontactpoints_pos, T_origin_newcontactpoints_orn, curr_contact_points):
        """
        TODO: make this private?
        :param i:
        :param T_origin_newcontactpoints:
        :return:
        """
        _, curr_obj_orn = self._sim.objects.get_curr_pose()
        T_cp_origin = p.invertTransform(curr_contact_points[i], curr_obj_orn)
        link = p.getLinkState(self._sim.hand.id, j)
        distal_pos = link[4]
        distal_orn = link[5]
        T_cp_link = p.multiplyTransforms(T_cp_origin[0], T_cp_origin[1], distal_pos, distal_orn)
        T_origin_nl = p.multiplyTransforms(T_origin_newcontactpoints_pos[i], T_origin_newcontactpoints_orn[i],
                                           T_cp_link[0], T_cp_link[1])

        return T_origin_nl

    def _get_pose_in_world_origin_expert(self, data):
        """
        TODO: make this private?
        Gets the new contact points in world coordinates
        :param cube: instance of object in scene class(object to move)
        :param data: line in file of human data [x,y,rmag,f_x,f_y,f_rot_mag]
        :return: list T_origin_newcontactpoints: next contact points in world coordinates for left and right
        """

        T_origin_nextpose_cube = self._get_origin_cube(data)
        curr_contact_points = self.get_contact_points(self._sim.objects.id)
        if None in curr_contact_points:
            return [None, None, self.maintain_contact(), None, None]
        obj_pos, obj_orn = self._sim.objects.get_curr_pose()
        T_cube_origin = p.invertTransform(obj_pos, obj_orn)
        T_origin_new_cp_pos = []
        T_origin_new_cp_orn = []
        T_origin_new_link_pos = []
        T_origin_new_link_orn = []
        for i in range(0, len(self._sim.hand.end_effector_indices)):
            T_origin_new_cp = self.get_origin_cp(i, self._sim.objects.id, T_cube_origin, T_origin_nextpose_cube,
                                                 curr_contact_points)
            # print("Contact")
            T_origin_new_cp_pos.append(T_origin_new_cp[0])
            T_origin_new_cp_orn.append(T_origin_new_cp[1])
            T_origin_nl = self.get_origin_links(i, self._sim.hand.end_effector_indices[i], T_origin_new_cp_pos,
                                                T_origin_new_cp_orn, curr_contact_points)
            T_origin_new_link_pos.append(T_origin_nl[0])
            T_origin_new_link_orn.append(T_origin_nl[1])

        return [T_origin_new_cp_pos, T_origin_new_cp_orn, T_origin_new_link_pos, T_origin_new_link_orn,
         T_origin_nextpose_cube]
        # return T_origin_new_link_pos

    def select_action(self):
        """
        This controller is designed ot move an object along a certain path
        :return: action
        """
        cube_next_pose = self.get_next_line()
        # print("Cube Pose: {}".format(cube_next_pose))
        next_info = self._get_pose_in_world_origin_expert(cube_next_pose)
        # print(next_info)
        next_contact_points = next_info[2]
        action = p.calculateInverseKinematics2(bodyUniqueId=self._sim.hand.id,
                                               endEffectorLinkIndices=self._sim.hand.end_effector_indices,
                                               targetPositions=next_contact_points)
        # if next_info[0] is not None:
        #     Markers.Marker().set_marker_pose(next_info[0][0])
        #     Markers.Marker().set_marker_pose(next_info[0][1])
        return action


import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

# Implementation of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971
# [Not the implementation used in the TD3 paper]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        torch.nn.init.kaiming_uniform_(self.l1.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        self.l2 = nn.Linear(400, 300)
        torch.nn.init.kaiming_uniform_(self.l2.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        self.l3 = nn.Linear(300, action_dim)
        torch.nn.init.kaiming_uniform_(self.l3.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.sigmoid(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 400)
        torch.nn.init.kaiming_uniform_(self.l1.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        self.l2 = nn.Linear(400, 300)
        torch.nn.init.kaiming_uniform_(self.l2.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        self.l3 = nn.Linear(300, 1)
        torch.nn.init.kaiming_uniform_(self.l3.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

        self.max_q_value = 50

    def forward(self, state, action):
        q = F.relu(self.l1(torch.cat([state, action], -1)))
        q = F.relu(self.l2(q))
        return self.max_q_value * torch.sigmoid(self.l3(q))
    # return self.l3(q)


class DDPGfD(ControllerBase):
    def __init__(self, state_path=None, state_dim=35, action_dim=4, max_action=2, n=5, discount=0.995, tau=0.0005, batch_size=64,
                 expert_sampling_proportion=0.7):
        super().__init__(state_path)
        self.dir = 'c'
        ControllerBase._sim.set_obj_target_pose(self.dir)
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-4, weight_decay=1e-4)

        self.discount = discount
        self.tau = tau
        self.n = n
        self.network_repl_freq = 10
        self.total_it = 0
        self.lambda_Lbc = 1

        # Sample from the expert replay buffer, decaying the proportion expert-agent experience over time
        self.initial_expert_proportion = expert_sampling_proportion
        self.current_expert_proportion = expert_sampling_proportion
        self.sampling_decay_rate = 0.2
        self.sampling_decay_freq = 400

        # Most recent evaluation reward produced by the policy within training
        self.avg_evaluation_reward = 0

        self.batch_size = batch_size

    def select_action(self):
        self.state.update()
        # print("TYPE:", type(self.state.get_obs()), self.state.get_obs())
        state = torch.FloatTensor(np.reshape(self.state.get_obs(), (1, -1))).to(device)
        # print("TYPE:", type(state), state)
        action = self.actor(state).cpu().data.numpy().flatten()
        print("Action: {}".format(action))
        return action

    def train(self, episode_step, expert_replay_buffer, replay_buffer=None, prob=0.7):
        """ Update policy based on full trajectory of one episode """
        self.total_it += 1

        # Determine which replay buffer to sample from
        if replay_buffer is not None and expert_replay_buffer is None:  # Only use agent replay
            expert_or_random = "agent"
        elif replay_buffer is None and expert_replay_buffer is not None:  # Only use expert replay
            expert_or_random = "expert"
        else:
            expert_or_random = np.random.choice(np.array(["expert", "agent"]), p=[prob, round(1. - prob, 2)])

        if expert_or_random == "expert":
            state, action, next_state, reward, not_done = expert_replay_buffer.sample()
        else:
            state, action, next_state, reward, not_done = replay_buffer.sample()

        """
		finger_reward_count = 0
		grasp_reward_count = 0
		lift_reward_count = 0
		non_zero_count = 0
		for elem in reward:
			if elem != 0:
				non_zero_count += 1
				if elem < 5:
					finger_reward_count += 1
				elif elem < 10:
					grasp_reward_count += 1
				elif elem >= 10:
					lift_reward_count += 1
		print("\nIN OG TRAIN: non_zero_reward: ",non_zero_count)
		print("IN OG TRAIN: finger_reward_count: ", finger_reward_count)
		print("IN OG TRAIN: grasp_reward_count: ", grasp_reward_count)
		print("IN OG TRAIN: lift_reward_count: ", lift_reward_count)
		"""

        # Target Q network
        # print("Target Q")
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        # print(target_Q.shape)
        # print("target_Q: ",target_Q)
        target_Q = reward + (self.discount * target_Q).detach()  # bellman equation
        # print(target_Q.shape)
        # print("target_Q: ",target_Q)

        # print("Target_QN")
        # Compute the target Q_N value
        rollreward = []
        target_QN = self.critic_target(next_state[(self.n - 1):], self.actor_target(next_state[(self.n - 1):]))
        # print(target_QN.shape)
        # print("target_QN: ",target_Q)

        ep_timesteps = episode_step
        if state.shape[0] < episode_step:
            ep_timesteps = state.shape[0]

        for i in range(ep_timesteps):
            if i >= (self.n - 1):
                roll_reward = (self.discount ** (self.n - 1)) * reward[i].item() + (self.discount ** (self.n - 2)) * \
                              reward[i - (self.n - 2)].item() + (self.discount ** 0) * reward[i - (self.n - 1)].item()
                rollreward.append(roll_reward)

        # print("After Calc len(rollreward): ",len(rollreward))
        # print("After Calc rollreward: ", rollreward)

        if len(rollreward) != ep_timesteps - (self.n - 1):
            raise ValueError

        rollreward = torch.FloatTensor(np.array(rollreward).reshape(-1, 1)).to(device)

        # print("After reshape len(rollreward): ",len(rollreward))
        # print("After reshape rollreward: ", rollreward)

        # Calculate target network
        target_QN = rollreward + (
                    self.discount ** self.n) * target_QN  # bellman equation <= this is the final N step return

        # print("Target QN")
        # print("Target_QN.shape: ",target_QN.shape)
        # print("Target_QN: ", target_QN)

        # Get current Q estimate
        current_Q = self.critic(state, action)

        # print("current_Q")
        # print("current_Q.shape: ",current_Q.shape)
        # print("current_Q: ", current_Q)

        # Yi's old implementation - not needed for loss calculation
        # current_Q_n = self.critic(state[:(ep_timesteps - (self.n - 1))], action[:(ep_timesteps - (self.n - 1))])

        # print("current_Q_n")
        # print("current_Q_n.shape: ",current_Q_n.shape)
        # print("current_Q_n: ", current_Q_n)

        # L_1 loss (Loss between current state, action and reward, next state, action)
        critic_L1loss = F.mse_loss(current_Q, target_Q)

        # print("critic_L1loss")
        # print("critic_L1loss.shape: ",critic_L1loss.shape)
        # print("critic_L1loss: ", critic_L1loss)

        # L_2 loss (Loss between current state, action and reward, n state, n action)
        critic_LNloss = F.mse_loss(current_Q, target_QN)

        # print("critic_LNloss")
        # print("critic_LNloss.shape: ",critic_LNloss.shape)
        # print("critic_LNloss: ", critic_LNloss)

        # Total critic loss
        lambda_1 = 0.5  # hyperparameter to control n loss
        critic_loss = critic_L1loss + lambda_1 * critic_LNloss

        # print("critic_loss")
        # print("critic_loss.shape: ", critic_loss.shape)
        # print("critic_loss: ", critic_loss)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()

        # print("actor_loss")
        # print("actor_loss.shape: ", actor_loss.shape)
        # print("actor_loss: ", actor_loss)

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.total_it % self.network_repl_freq == 0:
            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        return actor_loss.item(), critic_loss.item(), critic_L1loss.item(), critic_LNloss.item()

    def train_batch(self, max_episode_num, episode_num, update_count, expert_replay_buffer, replay_buffer):
        """ Update policy networks based on batch_size of episodes using n-step returns """
        self.total_it += 1
        agent_batch_size = 0
        expert_batch_size = 0

        # Sample replay buffer
        if replay_buffer is not None and expert_replay_buffer is None:  # Only use agent replay
            # print("AGENT")
            expert_or_random = "agent"
            agent_batch_size = self.batch_size
            state, action, next_state, reward, not_done = replay_buffer.sample_batch_nstep(self.batch_size)
        elif replay_buffer is None and expert_replay_buffer is not None:  # Only use expert replay
            # print("EXPERT")
            expert_or_random = "expert"
            expert_batch_size = self.batch_size
            state, action, next_state, reward, not_done = expert_replay_buffer.sample_batch_nstep(self.batch_size)
        else:
            # print("MIX OF AGENT AND EXPERT")

            # Calculate proportion of expert sampling based on decay rate -- only calculate on the first update (to avoid repeats)
            if (episode_num + 1) % self.sampling_decay_freq == 0 and update_count == 0:
                prop_w_decay = self.initial_expert_proportion * pow((1 - self.sampling_decay_rate),
                                                                    int((episode_num + 1) / self.sampling_decay_freq))
                self.current_expert_proportion = max(0, prop_w_decay)
                print(
                    "In proportion calculation, episode_num + 1: {}, prop_w_decay: {}, self.current_expert_proportion: {}".format(
                        episode_num + 1, prop_w_decay, self.current_expert_proportion))

            # Sample from the expert and agent replay buffers
            expert_batch_size = int(self.batch_size * self.current_expert_proportion)
            agent_batch_size = self.batch_size - expert_batch_size
            # Get batches from respective replay buffers
            # print("SAMPLING FROM AGENT...agent_batch_size: ",agent_batch_size)
            agent_state, agent_action, agent_next_state, agent_reward, agent_not_done = replay_buffer.sample_batch_nstep(
                agent_batch_size)
            # print("SAMPLING FROM EXPERT...expert_batch_size: ",expert_batch_size)
            expert_state, expert_action, expert_next_state, expert_reward, expert_not_done = expert_replay_buffer.sample_batch_nstep(
                expert_batch_size)

            # Concatenate batches of agent and expert experience to get batch_size tensors of experience
            state = torch.cat((torch.squeeze(agent_state), torch.squeeze(expert_state)), 0)
            action = torch.cat((torch.squeeze(agent_action), torch.squeeze(expert_action)), 0)
            next_state = torch.cat((torch.squeeze(agent_next_state), torch.squeeze(expert_next_state)), 0)
            reward = torch.cat((torch.squeeze(agent_reward), torch.squeeze(expert_reward)), 0)
            not_done = torch.cat((torch.squeeze(agent_not_done), torch.squeeze(expert_not_done)), 0)
            if self.batch_size == 1:
                state = state.unsqueeze(0)
                action = action.unsqueeze(0)
                next_state = next_state.unsqueeze(0)
                reward = reward.unsqueeze(0)
                not_done = not_done.unsqueeze(0)

        reward = reward.unsqueeze(-1)
        not_done = not_done.unsqueeze(-1)

        ### FOR TESTING:
        # assert_batch_size = self.batch_size * num_trajectories
        num_timesteps_sampled = len(reward)

        # assert_n_steps = 5
        # assert_mod_state_dim = 82

        # assert state.shape == (assert_batch_size, assert_n_steps, assert_mod_state_dim)
        # assert next_state.shape == (assert_batch_size, assert_n_steps, assert_mod_state_dim)
        # assert action.shape == (assert_batch_size, assert_n_steps, 4)
        # assert reward.shape == (assert_batch_size, assert_n_steps, 1)
        # assert not_done.shape == (assert_batch_size, assert_n_steps, 1)

        # print("Target Q")
        target_Q = self.critic_target(next_state[:, 0], self.actor_target(next_state[:, 0]))
        # assert target_Q.shape == (assert_batch_size, 1)

        # print("Target Q: ",target_Q)
        # If we're randomly sampling trajectories, we need to index based on the done signal
        num_trajectories = len(not_done[:, 0])
        for n in range(num_trajectories):
            if not_done[n, 0]:  # NOT done
                target_Q[n, 0] = reward[n, 0] + (self.discount * target_Q[n, 0]).detach()  # bellman equation
            else:  # Final episode trajectory reward value
                target_Q[n, 0] = reward[n, 0]

        # print(target_Q.shape)
        # print("target_Q: ",target_Q)
        # assert target_Q.shape == (assert_batch_size, 1)

        # print("Target action")
        target_action = self.actor_target(next_state[:, -1])
        # print(target_action.shape)
        # print("target_action: ", target_action)
        # assert target_action.shape == (assert_batch_size, 4)

        # print("Target Critic Q value")
        target_critic_val = self.critic_target(next_state[:, -1], target_action)  # shape: (self.batch_size, 1)
        # print(target_critic_val.shape)
        # print("target_critic_val: ",target_critic_val)
        # assert target_Q.shape == (assert_batch_size, 1)

        n_step_return = torch.zeros(num_timesteps_sampled).to(device)  # shape: (self.batch_size,)
        # print("N step return before calculation (N=5)")
        # print(n_step_return.shape)
        # print("n_step_return: ", n_step_return)
        # assert n_step_return.shape == (assert_batch_size,)

        for i in range(self.n):
            n_step_return += (self.discount ** i) * reward[:, i].squeeze(-1)

        # print("N step return after calculation (N=5)")
        # print(n_step_return.shape)
        # print("n_step_return: ", n_step_return)
        # assert n_step_return.shape == (assert_batch_size,)

        # print("Target QN, N STEPS")
        # this is the n step return with the added value fn estimation
        target_QN = n_step_return + (self.discount ** self.n) * target_critic_val.squeeze(-1)
        # print(target_QN.shape)
        # print("target_QN: ",target_QN)
        # assert target_QN.shape == (assert_batch_size,)
        target_QN = target_QN.unsqueeze(dim=-1)
        # print(target_QN.shape)
        # print("target_QN: ", target_QN)
        # assert target_QN.shape == (assert_batch_size, 1)

        # print("Current Q")
        # New implementation
        current_Q = self.critic(state[:, 0], action[:, 0])
        # print(current_Q.shape)
        # print("current_Q: ", current_Q)
        # assert current_Q.shape == (assert_batch_size, 1)

        # print("CRITIC L1 Loss:")
        # L_1 loss (Loss between current state, action and reward, next state, action)
        critic_L1loss = F.mse_loss(current_Q, target_Q)
        # print(critic_L1loss.shape)
        # print("critic_L1loss: ", critic_L1loss)

        # print("CRITIC LN Loss:")
        # L_2 loss (Loss between current state, action and reward, n state, n action)
        critic_LNloss = F.mse_loss(current_Q, target_QN)
        # print(critic_LNloss.shape)
        # print("critic_LNloss: ", critic_LNloss)

        # print("CRITIC Loss (L1 loss + lambda * LN Loss):")
        # Total critic loss
        lambda_1 = 0.5  # hyperparameter to control n loss
        critic_loss = critic_L1loss + lambda_1 * critic_LNloss
        # print(critic_loss.shape)
        # print("critic_loss: ", critic_loss)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute Behavior Cloning loss - state and action are from the expert
        Lbc = 0
        # If we are decaying the amount of expert experience, then decay the BC loss as well
        if self.sampling_decay_rate != 0:
            self.lambda_Lbc = self.current_expert_proportion
        # Compute loss based on Mean Squared Error between the actor network's action and the expert's action
        if expert_batch_size > 0:
            # Expert state and expert action are sampled from the expert demonstrations (expert replay buffer)
            Lbc = F.mse_loss(self.actor(expert_state), expert_action)
        # print("self.lambda_Lbc: ", self.lambda_Lbc)

        # Compute actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean() + self.lambda_Lbc * Lbc
        # print("Actor loss: ")
        # print(actor_loss.shape)
        # print("actor_loss: ",actor_loss)

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.total_it % self.network_repl_freq == 0:
            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        return actor_loss.item(), critic_loss.item(), critic_L1loss.item(), critic_LNloss.item()

    def copy(self, policy_to_copy_from):
        """ Copy input policy to be set to another policy instance
		policy_to_copy_from: policy that will be copied from
        """
        # Copy the actor and critic networks
        self.actor = copy.deepcopy(policy_to_copy_from.actor)
        self.actor_target = copy.deepcopy(policy_to_copy_from.actor_target)
        self.actor_optimizer = copy.deepcopy(policy_to_copy_from.actor_optimizer)

        self.critic = copy.deepcopy(policy_to_copy_from.critic)
        self.critic_target = copy.deepcopy(policy_to_copy_from.critic_target)
        self.critic_optimizer = copy.deepcopy(policy_to_copy_from.critic_optimizer)

        self.discount = policy_to_copy_from.discount
        self.tau = policy_to_copy_from.tau
        self.n = policy_to_copy_from.n
        self.network_repl_freq = policy_to_copy_from.network_repl_freq
        self.total_it = policy_to_copy_from.total_it
        self.lambda_Lbc = policy_to_copy_from.lambda_Lbc
        self.avg_evaluation_reward = policy_to_copy_from.avg_evaluation_reward

        # Sample from the expert replay buffer, decaying the proportion expert-agent experience over time
        self.initial_expert_proportion = policy_to_copy_from.initial_expert_proportion
        self.current_expert_proportion = policy_to_copy_from.current_expert_proportion
        self.sampling_decay_rate = policy_to_copy_from.sampling_decay_rate
        self.sampling_decay_freq = policy_to_copy_from.sampling_decay_freq
        self.batch_size = policy_to_copy_from.batch_size

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_target.state_dict(), filename + "_critic_target")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_target.state_dict(), filename + "_actor_target")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")
        np.save(filename + "avg_evaluation_reward", np.array([self.avg_evaluation_reward]))

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic", map_location=device))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer", map_location=device))
        self.actor.load_state_dict(torch.load(filename + "_actor", map_location=device))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer", map_location=device))

        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target = copy.deepcopy(self.actor)
