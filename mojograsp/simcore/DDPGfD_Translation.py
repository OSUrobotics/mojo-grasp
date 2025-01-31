#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 14:00:12 2022

@author: orochi
"""

# import Markers
import copy
from random import sample
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from DDPGfD import *
# Implementation of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971
# [Not the implementation used in the TD3 paper]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class AltCritic(nn.Module):
    def __init__(self, state_dim, action_dim, pred_dim):
        super(AltCritic, self).__init__()
        self.leaky = nn.LeakyReLU()
        self.l1 = nn.Linear(state_dim + action_dim, 100)
        torch.nn.init.kaiming_uniform_(self.l1.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        self.l2 = nn.Linear(100, 50)
        torch.nn.init.kaiming_uniform_(self.l2.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        self.l3 = nn.Linear(50, pred_dim)
        torch.nn.init.kaiming_uniform_(self.l3.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

        self.max_q_value = 0.1

    def forward(self, state, action):
        # print('input to critic', torch.cat([state, action], -1))
        state = simple_normalize(state)
        q = F.relu(self.l1(torch.cat([state, action], -1)))
        q = F.relu(self.l2(q))
        # print("Q Critic: {}".format(q))
        q = torch.tanh(self.l3(q))
        return q * self.max_q_value
    
class DDPGTranslate():
    def __init__(self, arg_dict: dict = None):
        # state_path=None, state_dim=32, action_dim=4, max_action=1.57, n=5, discount=0.995, tau=0.0005, batch_size=10,
        #          expert_sampling_proportion=0.7):
        if arg_dict is None:
            print('no arg dict')
            arg_dict = {'state_dim': 32, 'action_dim': 4, 'max_action': 1.57, 'n': 5, 'discount': 0.995, 'tau': 0.005,
                        'batch_size': 10, 'expert_sampling_proportion': 0.7, 'pred_dim': 2}
        self.state_dim = arg_dict['state_dim']
        self.action_dim = arg_dict['action_dim']
        self.pred_dim = arg_dict['pred_dim']
        self.actor = Actor(self.state_dim, self.action_dim, arg_dict['max_action']).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)


        self.critic = AltCritic(self.state_dim, self.action_dim, self.pred_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-4, weight_decay=1e-4)

        self.actor_loss = []
        self.critic_loss = []
        self.critic_L1loss = []
        self.critic_LNloss = []
        self.writer = SummaryWriter()

        self.discount = arg_dict['discount']
        self.tau = arg_dict['tau']
        self.n = arg_dict['n']
        self.network_repl_freq = 2
        self.total_it = 0
        self.lambda_Lbc = 1

        # Sample from the expert replay buffer, decaying the proportion expert-agent experience over time
        self.initial_expert_proportion = arg_dict['expert_sampling_proportion']
        self.current_expert_proportion = arg_dict['expert_sampling_proportion']
        self.sampling_decay_rate = 0.2
        self.sampling_decay_freq = 400

        # Most recent evaluation reward produced by the policy within training
        self.avg_evaluation_reward = 0

        self.batch_size = arg_dict['batch_size']
        self.rng = default_rng()
        self.rollout = True
        self.u_count = 0

    def select_action(self, state):
        state = torch.FloatTensor(np.reshape(state, (1, -1))).to(device)
        # print("TYPE:", type(state), state)
        action = self.actor(state).cpu().data.numpy().flatten()
        # print("Action: {}".format(action))
        return action

    def grade_action(self, state, action):
        state = torch.FloatTensor(np.reshape(state, (1,-1))).to(device)
        action = torch.FloatTensor(np.reshape(action, (1,-1))).to(device)
        grade = self.critic(state, action).cpu().data.numpy().flatten()
        return grade

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

    def update_target(self):
        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def calc_roll_rewards(self,rollout_reward):
        sum_rewards = []
        num_rewards = []
        for reward_set in rollout_reward:
            temp_reward = 0
            for count, step_reward in enumerate(reward_set):
                temp_reward += step_reward*self.discount**count
            sum_rewards.append(temp_reward)
            num_rewards.append(count+1)
        num_rewards = torch.tensor(num_rewards).to(device)
        num_rewards = torch.unsqueeze(num_rewards,1)
        sum_rewards = torch.tensor(sum_rewards)
        sum_rewards = torch.unsqueeze(sum_rewards,1)
        return sum_rewards, num_rewards

    def collect_batch(self, replay_buffer):
        #also change replay buffer termination to have nasty negative reward if we terminate early from failing
        # also talk to kegan about expert control
        num_timesteps = len(replay_buffer)
        if num_timesteps < self.batch_size:
            return None, None, None, None, None, None
        else:
            if self.rollout:
                sampled_data, sampled_rewards, sampled_last_state = replay_buffer.sample_rollout(self.batch_size, 5)
            else:
                sampled_data = replay_buffer.sample(self.batch_size)

            state = []
            action = []
            reward = []
            next_state = []
            goal_dir = []
            ops_dir = []
            for timestep in sampled_data:
                t_state = timestep.state
                temp_state = []
                temp_state.extend(t_state['obj_2']['pose'][0])
                temp_state.extend(t_state['obj_2']['pose'][1])
                temp_state.extend([item for item in t_state['two_finger_gripper']['joint_angles'].values()])
                temp_state.extend(timestep.reward['goal_position'])
                temp_state.extend([t_state['f1_obj_dist'],t_state['f2_obj_dist']])
                state.append(temp_state)
                action.append(timestep.action['target_joint_angles'])
                tstep_reward = timestep.reward['deltas']
                reward.append(tstep_reward)
                goal_direction = timestep.reward['v_to_goal']
                opposite_dir = [goal_direction[1],-goal_direction[0]]
                ops_dir.append(opposite_dir)
                goal_dir.append(goal_direction)
                t_next_state = timestep.next_state
                temp_next_state = []
                temp_next_state.extend(t_next_state['obj_2']['pose'][0])
                temp_next_state.extend(t_next_state['obj_2']['pose'][1])
                temp_next_state.extend([item for item in t_next_state['two_finger_gripper']['joint_angles'].values()])
                temp_next_state.extend(timestep.reward['goal_position'])
                temp_next_state.extend([t_next_state['f1_obj_dist'],t_next_state['f2_obj_dist']])
                next_state.append(temp_next_state)
            state = torch.tensor(state)
            action = torch.tensor(action)
            reward = torch.tensor(reward)
            reward = torch.unsqueeze(reward, 1)
            next_state = torch.tensor(next_state)
            goal_dir = torch.tensor(goal_dir)
            ops_dir = torch.tensor(ops_dir)
            state = state.to(device)
            action = action.to(device)
            next_state = next_state.to(device)
            reward = reward.to(device)
            goal_dir = goal_dir.to(device)
            ops_dir = ops_dir.to(device)
            return state, action, next_state, reward, goal_dir, ops_dir

    def train(self, expert_replay_buffer, replay_buffer=None, prob=0.7):
        """ Update policy based on full trajectory of one episode """
        self.total_it += 1

        # Determine which replay buffer to sample from
        if replay_buffer is not None and expert_replay_buffer is None:  # Only use agent replay
            returned_buffer = replay_buffer
        elif replay_buffer is None and expert_replay_buffer is not None:  # Only use expert replay
            returned_buffer = expert_replay_buffer
        else:
            returned_buffer = np.random.choice(np.array([expert_replay_buffer, replay_buffer]),
                                               p=[prob, round(1. - prob, 2)])

        state, action, next_state, reward, goal_dir, ops_dir = self.collect_batch(returned_buffer)
        if state is not None:
            # since reward is the deltas and we want to predict that, THAT is the target for the critic
            # as such, no roll rewards for you

            # Get current Q estimate
            current_Q = self.critic(state, action)

            # L_1 loss (Loss between current state, action and reward, next state, action)
            critic_loss = F.mse_loss(current_Q, torch.squeeze(reward))
            # print(current_Q.shape,torch.squeeze(reward).shape)

            # Total critic loss

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            step_estimation = self.critic(state, self.actor(state))
            # Compute actor loss
            # print(step_estimation.shape)
            # print(torch.tensordot(goal_dir, step_estimation, dims=([1],[1])).shape)
            actor_loss = -(torch.tensordot(goal_dir, step_estimation)-abs(torch.tensordot(ops_dir,step_estimation)) - 1)
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.writer.add_scalar('Loss/critic',critic_loss.detach(),self.total_it)
            self.writer.add_scalar('Loss/actor',actor_loss.detach(),self.total_it)


            # update target networks
            if self.total_it % self.network_repl_freq == 0:
                self.update_target()
                self.u_count +=1
                # print('updated ', self.u_count,' times')
                # print('critic of state and action')
                # print(self.critic(state, self.actor(state)))
                # print('target q - current q')
                # print(target_Q-current_Q)
            return actor_loss.item(), critic_loss.item(), 0, 0
        
class DDPGMultiTranslate(DDPGTranslate):
    def __init__(self, arg_dict: dict = None):
        # state_path=None, state_dim=32, action_dim=4, max_action=1.57, n=5, discount=0.995, tau=0.0005, batch_size=10,
        #          expert_sampling_proportion=0.7):
        if arg_dict is None:
            print('no arg dict')
            arg_dict = {'state_dim': 32, 'action_dim': 4, 'max_action': 1.57, 'n': 5, 'discount': 0.995, 'tau': 0.005,
                        'batch_size': 10, 'expert_sampling_proportion': 0.7, 'pred_dim': 2}
        try: 
            self.num_actors =arg_dict['num_actors']
        except:
            self.num_actors = 8
        self.state_dim = arg_dict['state_dim']
        self.action_dim = arg_dict['action_dim']
        self.pred_dim = arg_dict['pred_dim']
        self.actor = [Actor(self.state_dim, self.action_dim, arg_dict['max_action']).to(device) for _ in range(self.num_actors)]
        self.actor_target = [copy.deepcopy(i) for i in self.actor]
        self.actor_optimizer = [torch.optim.Adam(i.parameters(), lr=1e-4) for i in self.actor]


        self.critic = AltCritic(self.state_dim, self.action_dim, self.pred_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-4, weight_decay=1e-4)

        self.actor_loss = []
        self.critic_loss = []
        self.critic_L1loss = []
        self.critic_LNloss = []
        self.writer = SummaryWriter()

        self.discount = arg_dict['discount']
        self.tau = arg_dict['tau']
        self.n = arg_dict['n']
        self.network_repl_freq = 2
        self.total_it = 0
        self.lambda_Lbc = 1

        # Sample from the expert replay buffer, decaying the proportion expert-agent experience over time
        self.initial_expert_proportion = arg_dict['expert_sampling_proportion']
        self.current_expert_proportion = arg_dict['expert_sampling_proportion']
        self.sampling_decay_rate = 0.2
        self.sampling_decay_freq = 400

        # Most recent evaluation reward produced by the policy within training
        self.avg_evaluation_reward = 0

        self.batch_size = arg_dict['batch_size']
        self.rng = default_rng()
        self.rollout = True
        self.u_count = 0

    def train(self, replay_buffer, actor_num):
        """ Update policy based on batch of timesteps """
        self.total_it += 1

        state, action, next_state, reward, rollout_reward, last_state = self.collect_batch(replay_buffer)
        if state is not None:
        # print(sampled_data['next_state'])
        # print(torch.tensor(sampled_data['next_state']))
    
            target_Q = self.critic_target(next_state, self.actor_target[actor_num]((next_state)))
    
            target_Q = reward + (self.discount * target_Q).detach()  # bellman equation
    
            target_Q = target_Q.float()
    
            # Compute the roll rewards and the number of steps forward (could be less than rollout size if timestep near end of trial)
            sum_rewards, num_rewards = self.calc_roll_rewards(rollout_reward)
    
            target_QN = self.critic_target(last_state, self.actor_target[actor_num](last_state))
    
            # Compute QN from roll reward and discounted final state
            target_QN = sum_rewards.to(device) + (self.discount**num_rewards * target_QN).detach()
    
            target_QN = target_QN.float()
    
            # Get current Q estimate
            current_Q = self.critic(state, action)
    
            # L_1 loss (Loss between current state, action and reward, next state, action)
            critic_L1loss = F.mse_loss(current_Q, target_Q)
    
            # L_2 loss (Loss between current state, action and reward, n state, n action)
            critic_LNloss = F.mse_loss(current_Q, target_QN)
    
            # Total critic loss
            lambda_1 = 1  # hyperparameter to control n loss
            critic_loss = critic_L1loss.float() + lambda_1 * critic_LNloss
    
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
    
            # Compute actor loss
            actor_loss = -self.critic(state, self.actor[actor_num](state))
            priorities = 0.0001 + actor_loss**2 + 2*(current_Q-target_Q)**2
            actor_loss = actor_loss.mean()
    
            # Optimize the actor
            self.actor_optimizer[actor_num].zero_grad()
            actor_loss.backward()
            self.actor_optimizer[actor_num].step()
    
            self.writer.add_scalar('Loss/critic',critic_loss.detach(),self.total_it)
            self.writer.add_scalar('Loss/critic_L1',critic_L1loss.detach(),self.total_it)
            self.writer.add_scalar('Loss/critic_LN',critic_LNloss.detach(),self.total_it)
            self.writer.add_scalar('Loss/actor',actor_loss.detach(),self.total_it)
    
            
    
            # update target networks
            if self.total_it % self.network_repl_freq == 0:
                self.update_target()

            return actor_loss.item(), critic_loss.item(), critic_L1loss.item(), critic_LNloss.item()