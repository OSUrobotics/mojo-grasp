# import Markers
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy.random import default_rng


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
        # print("State Actor: {}\n{}".format(state.shape, state))
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        # return self.max_action * torch.sigmoid(self.l3(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 400)
        torch.nn.init.kaiming_uniform_(self.l1.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        self.l2 = nn.Linear(400, 300)
        torch.nn.init.kaiming_uniform_(self.l2.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        self.l3 = nn.Linear(300, 1)
        torch.nn.init.kaiming_uniform_(self.l3.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

        self.max_q_value = 40

    def forward(self, state, action):
        q = F.relu(self.l1(torch.cat([state, action], -1)))
        q = F.relu(self.l2(q))
        # print("Q Critic: {}".format(q))
        q = torch.tanh(self.l3(q))
        return self.max_q_value * q


class DDPGfD():
    def __init__(self, arg_dict: dict = None):
        # state_path=None, state_dim=32, action_dim=4, max_action=1.57, n=5, discount=0.995, tau=0.0005, batch_size=10,
        #          expert_sampling_proportion=0.7):
        if arg_dict is None:
            print('dict was none')
            arg_dict = {'state_dim': 32, 'action_dim': 4, 'max_action': 1.57, 'n': 5, 'discount': 0.995, 'tau': 0.0005,
                        'batch_size': 10, 'expert_sampling_proportion': 0.7}
        self.state_dim = arg_dict['state_dim']
        self.action_dim = arg_dict['action_dim']
        print('STATE DIM', self.state_dim)
        self.actor = Actor(self.state_dim, self.action_dim, arg_dict['max_action']).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(self.state_dim, self.action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-4, weight_decay=1e-4)

        self.actor_loss = 0
        self.critic_loss = 0
        self.critic_L1loss = 0
        self.critic_LNloss = 0

        self.discount = arg_dict['discount']
        self.tau = arg_dict['tau']
        self.n = arg_dict['n']
        self.network_repl_freq = 10
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


    def select_action(self, state):
        state = torch.FloatTensor(np.reshape(state, (1, -1))).to(device)
        # print("TYPE:", type(state), state)
        action = self.actor(state).cpu().data.numpy().flatten()
        # print("Action: {}".format(action))
        return action

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

    def collect_batch(self, replay_buffer):
        num_timesteps = len(replay_buffer)
        if num_timesteps < self.batch_size:
            return None, None, None, None
        else:
            sampled_data = replay_buffer.sample(self.batch_size)
            # print(len(sampled_data))
            state = []
            action = []
            reward = []
            next_state = []
            for timestep in sampled_data:
                t_state = timestep.state
                temp_state = []
                temp_state.extend(t_state['obj_2']['pose'][0])
                temp_state.extend(t_state['obj_2']['pose'][1])
                temp_state.extend([item for item in t_state['two_finger_gripper']['joint_angles'].values()])
                temp_state.extend(timestep.reward['goal_position'])
                state.append(temp_state)
                action.append(timestep.action['target_joint_angles'])
                reward.append(timestep.reward['distance_to_goal'])
                t_next_state = timestep.next_state
                temp_next_state = []
                temp_next_state.extend(t_next_state['obj_2']['pose'][0])
                temp_next_state.extend(t_next_state['obj_2']['pose'][1])
                temp_next_state.extend([item for item in t_next_state['two_finger_gripper']['joint_angles'].values()])
                temp_next_state.extend(timestep.reward['goal_position'])
                next_state.append(temp_next_state)
            state = torch.tensor(state)
            action = torch.tensor(action)
            reward = torch.tensor(reward)
            reward = torch.unsqueeze(reward,1)
            next_state = torch.tensor(next_state)
            state = state.to(device)
            action = action.to(device)
            next_state = next_state.to(device)
            reward = reward.to(device)
            # print(reward.shape)
            # print(reward)
            return state, action, next_state, reward

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

        state, action, next_state, reward = self.collect_batch(returned_buffer)
        not_done = True
        # print(state)
        # Target Q network
        # print("Target Q")

        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        # print(target_Q.shape)
        # print("target_Q: ",target_Q)
        target_Q = reward + (self.discount * target_Q).detach()  # bellman equation
        # print(target_Q.shape)
        # print("target_Q: ",target_Q)
        # print("target_Q.type: ", target_Q.type())
        target_Q = target_Q.float()
        # print("Target_QN")
        # Compute the target Q_N value
        rollreward = []
        target_QN = self.critic_target(next_state[(self.n - 1):], self.actor_target(next_state[(self.n - 1):]))
        # print(target_QN.shape)
        # print("target_QN: ", target_QN)

        # # Original:
        # ep_timesteps = episode_step
        ep_timesteps = 100  # episode_step

        # @ask_steph
        if state.shape[0] < ep_timesteps:
            ep_timesteps = state.shape[0]

        for i in range(ep_timesteps):
            if i >= (self.n - 1):
                # Original:
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
        #
        # print("Target QN")
        # print("Target_QN.shape: ", target_QN.type())
        # print("Target_QN.type: ", target_QN.type())
        # print("Target_QN: ", target_QN)

        # Get current Q estimate
        current_Q = self.critic(state, action)

        # print("current_Q")
        # print("current_Q.shape: ", current_Q.shape)
        # print("current_Q.type: ", current_Q.type())
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
        # critic_LNloss = F.mse_loss(current_Q, target_QN)

        # print("critic_LNloss")
        # print("critic_LNloss.shape: ",critic_LNloss.shape)
        # print("critic_LNloss: ", critic_LNloss)

        # Total critic loss
        lambda_1 = 0.5  # hyperparameter to control n loss
        critic_loss = critic_L1loss.float() #+ lambda_1 * critic_LNloss

        # print("critic_loss")
        # print("critic_loss.shape: ", critic_loss.shape)
        print("critic_loss: ", critic_loss)

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
        return actor_loss.item(), critic_loss.item(), critic_L1loss.item()#, critic_LNloss.item()

    def train_single_timestep(self, episode_step, expert_replay_buffer, replay_buffer=None, prob=0.7):
        """ Update policy based on full trajectory of one episode """
        self.total_it += 1

        # Determine which replay buffer to sample from
        if replay_buffer is not None and expert_replay_buffer is None:  # Only use agent replay
            expert_or_random = replay_buffer
        elif replay_buffer is None and expert_replay_buffer is not None:  # Only use expert replay
            expert_or_random = expert_replay_buffer
        else:
            expert_or_random = np.random.choice(np.array([expert_replay_buffer, replay_buffer]),
                                                p=[prob, round(1. - prob, 2)])

        returned_buffer = expert_or_random.get_between_timestep_random_sample(num_timesteps=1, start_timestep=42)
        state, action, next_state, reward = self.get_items_from_buffer(returned_buffer)
        not_done = True

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

        # # Original:
        # ep_timesteps = episode_step
        ep_timesteps = 1  # episode_step

        # @ask_steph
        if state.shape[0] < episode_step:
            ep_timesteps = state.shape[0]

        for i in range(ep_timesteps):
            if i >= (self.n - 1):
                roll_reward = (self.discount ** (self.n - 1)) * reward[i].item()
                # # Original:
                # roll_reward = (self.discount ** (self.n - 1)) * reward[i].item()  + (self.discount ** (self.n - 2)) * \
                #               reward[i - (self.n - 2)].item() + (self.discount ** 0) * reward[i - (self.n - 1)].item()

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
