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
# import os

# Implementation of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971
# [Not the implementation used in the TD3 paper]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def simple_normalize(x_tensor):
    # order is pos, orientation (quaternion), joint angles, velocity
    # maxes = torch.tensor([0.2, 0.35, 0.1, 1, 1, 1, 1, np.pi/2, 0, np.pi/2, np.pi, 1, 1, 1, 0.5, 0.5, 1, 1, 1, 1, 1, 1]).to(device)
    # mins = torch.tensor([-0.2, -0.05, 0.0, 0, 0, 0, 0,-np.pi/2, -np.pi, -np.pi/2, 0, 0, 0, 0, -0.01, -0.01, -1, -1, -1, -1, -1, -1]).to(device)
    # maxes = torch.tensor([0.2, 0.35, np.pi/2, 0, np.pi/2, np.pi, 1, 1, 0.2, 0.2, 0.2, 0.35, 0.2, 0.35, 0.055, 0.055]).to(device)
    # mins = torch.tensor([-0.2, -0.05,-np.pi/2, -np.pi, -np.pi/2, 0, -1, -1, -0.01, -0.01, -0.2, -0.05, -0.2, -0.05, -0.055, -0.055]).to(device)
    maxes = torch.tensor([0.072, 0.232, 0.072, 0.232, 0.072, 0.232, 0.072, 0.232]).to(device)
    mins = torch.tensor([-0.072, 0.088, -0.072, 0.088, -0.072, 0.088, -0.072, 0.088]).to(device)
    maxes = torch.tensor([0.072, 0.232, 0.072, 0.232,0.072, 0.232, 0.072, 0.232, 0.072, 0.232, 0.072, 0.232]).to(device)
    mins = torch.tensor([-0.072, 0.088, -0.072, 0.088, -0.072, 0.088, -0.072, 0.088, -0.072, 0.088, -0.072, 0.088]).to(device)
    y_tensor = ((x_tensor-mins)/(maxes-mins)-0.5) *2
    # print('normalized tensor', y_tensor)
    return y_tensor
    

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
        # print("State Actor: {}\n{}".format(state.shape, state))
        state = simple_normalize(state)
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        # return self.max_action * torch.sigmoid(self.l3(a))
        # return self.max_action * torch.tanh(self.l3(a))
        return torch.tanh(self.l3(a))
        # return self.l3(a)

# OLD PARAMS WERE 400-300, TESTING 100-50
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.leaky = nn.LeakyReLU()
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        torch.nn.init.kaiming_uniform_(self.l1.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        self.l2 = nn.Linear(400, 300)
        torch.nn.init.kaiming_uniform_(self.l2.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        self.l3 = nn.Linear(300, 1)
        torch.nn.init.kaiming_uniform_(self.l3.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

        self.max_q_value = 1

    def forward(self, state, action):
        # print('input to critic', torch.cat([state, action], -1))
        state = simple_normalize(state)
        q = F.relu(self.l1(torch.cat([state, action], -1)))
        q = F.relu(self.l2(q))
        # print("Q Critic: {}".format(q))
        # q = -torch.sigmoid(self.l3(q))
        # q = torch.tanh(self.l3(q))
        q = self.l3(q)
        return q# * self.max_q_value

class DDPGfD_priority():
    def __init__(self, arg_dict: dict = None):
        # state_path=None, state_dim=32, action_dim=4, max_action=1.57, n=5, discount=0.995, tau=0.0005, batch_size=10,
        #          expert_sampling_proportion=0.7):
        
        if arg_dict is None:
            print('no arg dict')
            arg_dict = {'state_dim': 32, 'action_dim': 4, 'max_action': 1.57, 'n': 5, 'discount': 0.995, 'tau': 0.005,
                        'batch_size': 20, 'tname':'test1'}
        self.state_dim = arg_dict['state_dim']
        print('Saving to tensorboard file', arg_dict['tname'])
        self.action_dim = arg_dict['action_dim']
        self.actor = Actor(self.state_dim, self.action_dim, arg_dict['max_action']).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4, weight_decay=1e-4)

        self.critic = Critic(self.state_dim, self.action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-4, weight_decay=1e-4)

        self.actor_loss = []
        self.critic_loss = []
        self.critic_L1loss = []
        self.critic_LNloss = []
        if arg_dict['reward'] == 'Sparse':
            self.sparse_reward = True
        else:
            self.sparse_reward = False
            
        if 'HER' in arg_dict['model']:
            self.use_HER = True
        else:
            self.use_HER = False
        print('USING SPARSE REWARDS:', self.sparse_reward)
        # TensorboardName = 'expert_trimmed'

        self.writer = SummaryWriter(arg_dict['tname'])

        self.discount = arg_dict['discount']
        self.tau = arg_dict['tau']
        self.n = arg_dict['n']
        self.network_repl_freq = 2
        self.total_it = 0
        self.lambda_Lbc = 1

        # Most recent evaluation reward produced by the policy within training
        self.avg_evaluation_reward = 0

        self.batch_size = arg_dict['batch_size']
        self.rng = default_rng()
        self.rollout = True
        self.u_count = 0
        
        self.actor_component = 10000
        self.critic_component = 1

    def select_action(self, state):
        state = torch.FloatTensor(np.reshape(state, (1, -1))).to(device)
        # print("TYPE:", type(state), state)
        action = self.actor(state).cpu().data.numpy().flatten()
        # print("Action: {}".format(action))
        return action

    def grade_action(self, state, action):
        state = torch.tensor(np.reshape(state, (1,-1)), dtype=float).to(device)
        action = torch.tensor(np.reshape(action, (1,-1)), dtype=float, requires_grad=True).to(device)
        state=state.float()
        action=action.float()
        action.retain_grad()
        g = self.critic(state, action)
        g.backward()
        grade = g.cpu().data.numpy().flatten()
        
        # print(action.grad)
        return grade, action.grad.cpu().data.numpy()
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
        types = []
        #TODO: normalize the rewards within the batch to -1,1
        #also change replay buffer termination to have nasty negative reward if we terminate early from failing
        # also talk to kegan about expert control
        # print('aaaaaAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')
        num_timesteps = len(replay_buffer)
        # print(num_timesteps)
        if num_timesteps < self.batch_size * 20:
            return None, None, None, None, None, None, None, None, None, None
        else:
            if self.rollout:
                sampled_data, transition_weight, indxs = replay_buffer.sample_rollout(self.batch_size, 5)
            else:
                sampled_data = replay_buffer.sample(self.batch_size)

            state = []
            action = []
            reward = []
            next_state = []
            rollout_reward = []
            last_state = []
            rollout_discount = []
            expert_status = []
            for i, timestep_series in enumerate(sampled_data):
                if len(timestep_series) > 0:
                    rtemp = 0
                    timestep = timestep_series[0]
                    t_state = timestep[0]
                    temp_state = []
                    temp_state.extend(t_state['obj_2']['pose'][0][0:2])
                    # temp_state.extend(t_state['obj_2']['pose'][1])
                    # temp_state.extend([item for item in t_state['two_finger_gripper']['joint_angles'].values()])
                    # temp_state.extend(t_state['obj_2']['velocity'][0][0:2])
                    # temp_state.extend([t_state['f1_obj_dist'],t_state['f2_obj_dist']])
                    temp_state.extend(t_state['f1_pos'][0:2])
                    temp_state.extend(t_state['f2_pos'][0:2])               
                    temp_state.extend(t_state['f1_base'][0:2])
                    temp_state.extend(t_state['f2_base'][0:2])         
                    temp_state.extend(timestep[2]['goal_position'][0:2])
                    state.append(temp_state)
                    action.append(timestep[1]['actor_output'])
                    # print(temp_state.type)
                    # print(timestep[1]['actor_output'].type)
                    if self.sparse_reward:
                        tstep_reward = timestep[2]['distance_to_goal'] < 0.002
                    else:
                        tstep_reward = max(-timestep[2]['distance_to_goal'] \
                            - max(timestep[2]['f1_dist'],timestep[2]['f2_dist'])/5,-1)
                    # reward.append(-timestep_series.reward['distance_to_goal'])
                    # tstep_reward = timestep[2]['distance_to_goal'] < 0.002
    
                    reward.append(tstep_reward)
                    t_next_state = timestep[3]
                    temp_next_state = []
                    # print(t_next_state)
                    temp_next_state.extend(t_next_state['obj_2']['pose'][0][0:2])
                    # temp_next_state.extend(t_next_state['obj_2']['pose'][1])
                    # temp_next_state.extend([item for item in t_next_state['two_finger_gripper']['joint_angles'].values()])
                    # temp_next_state.extend(t_next_state['obj_2']['velocity'][0][0:2])
                    # temp_next_state.extend([t_next_state['f1_obj_dist'],t_next_state['f2_obj_dist']])
                    temp_next_state.extend(t_next_state['f1_pos'][0:2])
                    temp_next_state.extend(t_next_state['f2_pos'][0:2])
                    temp_next_state.extend(t_next_state['f1_base'][0:2])
                    temp_next_state.extend(t_next_state['f2_base'][0:2])
                    temp_next_state.extend(timestep[2]['goal_position'][0:2])
                    next_state.append(temp_next_state)
                    expert_status.append(timestep[-1])
                    if self.rollout:
                        j =0
                        for j, timestep in enumerate(timestep_series[1:]):
                            if self.sparse_reward:
                                rtemp += (timestep[2]['distance_to_goal'] < 0.002) * self.discount ** (j+1)
                            else:
                                rtemp += -timestep[2]['distance_to_goal'] - max(timestep[2]['f1_dist'],timestep[2]['f2_dist'])/5* self.discount ** (j+1)
                            # 
                        rtemp = max(rtemp, -1)
                        temp_last_state = []
                        t_last_state = timestep_series[-1][0]
                        rollout_discount.append(j+1)
                        temp_last_state.extend(t_last_state['obj_2']['pose'][0][0:2])
                        # temp_last_state.extend(t_last_state['obj_2']['pose'][1])
                        # temp_last_state.extend(
                        # [item for item in t_last_state['two_finger_gripper']['joint_angles'].values()])
                        # temp_last_state.extend(t_last_state['obj_2']['velocity'][0][0:2])
                        # temp_last_state.extend([t_last_state['f1_obj_dist'],t_last_state['f2_obj_dist']])
                        temp_last_state.extend(t_last_state['f1_pos'][0:2])
                        temp_last_state.extend(t_last_state['f2_pos'][0:2])
                        temp_last_state.extend(t_last_state['f1_base'][0:2])
                        temp_last_state.extend(t_last_state['f2_base'][0:2])
                        temp_last_state.extend(timestep_series[0][2]['goal_position'][0:2])
                        last_state.append(temp_last_state)
                        rollout_reward.append(rtemp)
            state = torch.tensor(state)
            # print(reward)
            # print('state type', state.type)
            action = torch.tensor(action)
            # print('action type', action.type)
            action = action.float()
            reward = torch.tensor(reward)
            reward = torch.unsqueeze(reward, 1)
            next_state = torch.tensor(next_state)
            rollout_reward = torch.tensor(rollout_reward)
            rollout_reward = torch.unsqueeze(rollout_reward, 1)
            rollout_discount = torch.tensor(rollout_discount)
            rollout_discount = torch.unsqueeze(rollout_discount, 1)
            expert_status = torch.tensor(expert_status)
            expert_status = torch.unsqueeze(expert_status, 1)
            last_state = torch.tensor(last_state)
            state = state.to(device)
            action = action.to(device)
            next_state = next_state.to(device)
            reward = reward.to(device)
            rollout_reward = rollout_reward.to(device)
            rollout_discount = rollout_discount.to(device)
            expert_status = expert_status.to(device)
            last_state = last_state.to(device)
            # print(expert_status.shape)
            trimmed_weight = []
            trimmed_idxs = []
            
            # print('bruh')
            for tw, inds in zip(transition_weight, indxs):
                if len(tw) > 0:
                    # try:
                    #     temp = tw[0].tolist()
                    #     # print('wasnt a float')
                    #     # print(tw)
                    # except AttributeError:
                    #     print('was already a float')
                    #     print(tw)
                    #     temp = tw[0]
                    trimmed_weight.append(tw[0]) 
                    # print(trimmed_weight)
                    trimmed_idxs.append(inds[0])
            trimmed_weight = torch.tensor(trimmed_weight)
            trimmed_weight = torch.unsqueeze(trimmed_weight, 1)
            trimmed_weight = trimmed_weight.to(device)
            # print(rollout_reward)
            # print(rollout_discount)
            return state, action, next_state, reward, rollout_reward, rollout_discount, last_state, trimmed_weight, trimmed_idxs, expert_status

    def train(self, replay_buffer, prob=0.7):
        """ Update policy based on full trajectory of one episode """
        self.total_it += 1

        state, action, next_state, reward, sum_rewards, num_rewards, last_state, transition_weight, indxs, expert_status = self.collect_batch(replay_buffer)
        if state is not None:
            target_Q = self.critic_target(next_state, self.actor_target(next_state))

            target_Q = reward + (self.discount * target_Q).detach()  # bellman equation

            target_Q = target_Q.float()

            # Compute the roll rewards and the number of steps forward (could be less than rollout size if timestep near end of trial)
            target_QN = self.critic_target(last_state, self.actor_target(last_state))

            # Compute QN from roll reward and discounted final state
            target_QN = sum_rewards.to(device) + (self.discount**num_rewards * target_QN).detach()

            target_QN = (target_QN/num_rewards).float()

            # Get current Q estimate
            current_Q = self.critic(state, action)

            scaled_Q = current_Q * transition_weight
            
            scaled_target = target_Q * transition_weight
            
            scaled_QN = target_QN * transition_weight
            
            # L_1 loss (Loss between current state, action and reward, next state, action)
            # critic_L1loss = F.mse_loss(current_Q, target_Q)
            critic_L1loss = F.mse_loss(scaled_Q, scaled_target)

            # L_2 loss (Loss between current state, action and reward, n state, n action)
            # critic_LNloss = F.mse_loss(current_Q, target_QN)
            critic_LNloss = F.mse_loss(scaled_Q, scaled_QN)


            # Total critic loss
            lambda_1 = 0.5  # hyperparameter to control n loss
            critic_loss = critic_L1loss.float() + lambda_1 * critic_LNloss

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_value_(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()

            # Compute actor loss
            actor_action = self.actor(state)
            actor_action.retain_grad()
            individual_actor_loss = -self.critic(state, actor_action)
            # input(actor_loss)
            # print('actor_output', self.actor(state).shape)
            # print(action.shape)
            # print(state.shape)
            # print(state)
            
            
            
            # print('average without expert status', torch.mean(0.0001 + 0.25*individual_actor_loss**2 + 2*(current_Q-target_Q)**2))
            # print('percent expert', torch.mean(expert_status.float()))
            # print('start')
            # print(0.25*(individual_actor_loss)**2)
            # print(2*(current_Q[0]-target_Q)**2)
            # print(priorities)

            actor_loss = individual_actor_loss.mean()
            # temp = individual_actor_loss.shape
            # priorities = np.ones(temp)
            # Optimize the actor
            # TODO check the new priority method and make sure its sound
            priorities = expert_status*0.5 + 0.5

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            # print(actor_action.grad)
            # actor_component = actor_action.grad.mean(1,True)
            # print(actor_component.shape)
            # print(expert_status.shape)
            # cpart = (current_Q-target_Q)**2
            # apart = actor_component**2
            # priorities = expert_status*0.5 + 0.0001 + self.actor_component*apart + self.critic_component*cpart
            # print('actor portion', (2500000*actor_component**2).mean())
            
            # self.critic_component = 0.001 * 0.5/cpart.mean() + 0.999 * self.critic_component
            # self.actor_component = 0.001 * 0.5/apart.mean() + 0.999 * self.actor_component
            # print('actor part', self.actor_component)
            # print('critic part', self.critic_component)
            priorities = priorities.cpu().detach().numpy()
            nn.utils.clip_grad_value_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()

            self.writer.add_scalar('Loss/critic',critic_loss.detach(),self.total_it)
            self.writer.add_scalar('Loss/critic_L1',critic_L1loss.detach(),self.total_it)
            self.writer.add_scalar('Loss/critic_LN',critic_LNloss.detach(),self.total_it)
            self.writer.add_scalar('Loss/actor',actor_loss.detach(),self.total_it)

            replay_buffer.update_priorities(indxs,priorities)
            # update target networks
            if self.total_it % self.network_repl_freq == 0:
                self.update_target()
                self.u_count +=1
                # print('updated ', self.u_count,' times')
                # print('critic of state and action')
                # print(self.critic(state, self.actor(state)))
                # print('target q - current q')
                # print(target_Q-current_Q)
            return actor_loss.item(), critic_loss.item(), critic_L1loss.item(), critic_LNloss.item()



def unpack_arr(long_arr):
    """
    Unpacks an array of shape N x M x ... into array of N*M x ...
    :param: long_arr - array to be unpacked"""
    new_arr = [item for sublist in long_arr for item in sublist]
    return new_arr