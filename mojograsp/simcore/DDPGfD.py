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
from mojograsp.simcore.priority_replay_buffer import ReplayBufferPriority
from mojograsp.simcore.state import State
from mojograsp.simcore.reward import Reward
from csv import writer
import pickle as pkl
from torchmetrics import ExplainedVariance
# Implementation of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971
# [Not the implementation used in the TD3 paper]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert torch.cuda.is_available()
@torch.jit.script
def simple_normalize(x_tensor, mins, maxes):
    """
    normalizes a numpy array to -1 and 1 using provided maximums and minimums
    :param x_tensor: - array to be normalized
    :param mins: - array containing minimum values for the parameters in x_tensor
    :param maxes: - array containing maximum values for the parameters in x_tensor
    """
    # print(x_tensor.shape)
    return ((x_tensor-mins)/(maxes-mins)-0.5) *2

# @torch.jit.script    
def simple_normalize_noise(x_tensor, mins, maxes, noise_percent):
    """
    normalizes a numpy array to -1 and 1 using provided maximums and minimums
    :param x_tensor: - array to be normalized
    :param mins: - array containing minimum values for the parameters in x_tensor
    :param maxes: - array containing maximum values for the parameters in x_tensor
    """
    return ((x_tensor-mins)/(maxes-mins)-0.5) *2 + torch.normal(0,noise_percent, size=x_tensor.shape).to(device)

def unpack_arr(long_arr):
    """
    Unpacks an array of shape N x M x ... into array of N*M x ...
    :param: long_arr - array to be unpacked"""
    new_arr = [item for sublist in long_arr for item in sublist]
    return new_arr


class Actor(nn.Module):
    def __init__(self, state_dim:int, action_dim:int, max_action:float):
        """
        Constructor initializes actor network with input dimension 'state_dim' 
        and output dimension 'action_dim'. State mins and maxes saved for normalization
        """
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 400)
        torch.nn.init.kaiming_uniform_(self.l1.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        self.l2 = nn.Linear(400, 300)
        torch.nn.init.kaiming_uniform_(self.l2.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        self.l3 = nn.Linear(300, action_dim)
        torch.nn.init.kaiming_uniform_(self.l3.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        self.max_action = max_action

    def forward(self, state:torch.Tensor):
        """
        Runs state through actor network to get action associated with state
        """
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return torch.tanh(self.l3(a))

class Critic(nn.Module):
    def __init__(self, state_dim:int, action_dim:int):
        """
        Constructor initializes actor network with input dimension 'state_dim'+'action_dim' 
        and output dimension 1. State mins and maxes saved for normalization
        """
        super(Critic, self).__init__()
        self.leaky = nn.LeakyReLU()
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        torch.nn.init.kaiming_uniform_(self.l1.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        self.l2 = nn.Linear(400, 300)
        torch.nn.init.kaiming_uniform_(self.l2.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        self.l3 = nn.Linear(300, 1)
        torch.nn.init.kaiming_uniform_(self.l3.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

        self.max_q_value = 1

    def forward(self, state:torch.Tensor, action:torch.Tensor):
        """
        Concatenates state and action and runs through critic network to return
        q-value associated with state-action pair
        """
        q = F.relu(self.l1(torch.cat([state, action], -1)))
        q = F.relu(self.l2(q))
        q = self.l3(q)
        return q# * self.max_q_value

class DDPGfD_priority():
    def __init__(self, arg_dict: dict):
        """
        Constructor initializes the actor and critic network and use an argument
        dictionary to initialize hyperparameters
        """
        self.STATE_DIM = arg_dict['state_dim']
        self.REWARD_TYPE = arg_dict['reward']
        self.SAMPLING_STRATEGY = arg_dict['sampling']
        self.LAMBDA_1 = arg_dict['rollout_weight']
        self.ROLLOUT_SIZE = arg_dict['rollout_size']
        self.STATE_NOISE = arg_dict['state_noise']
        print('Saving to tensorboard file', arg_dict['tname'])
        self.ACTION_DIM = arg_dict['action_dim']
        self.actor = Actor(self.STATE_DIM, self.ACTION_DIM, arg_dict['max_action']).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=arg_dict['learning_rate'], weight_decay=1e-4)

        self.critic = Critic(self.STATE_DIM, self.ACTION_DIM).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=arg_dict['learning_rate'], weight_decay=1e-4)
        
        self.MAXES = torch.tensor(arg_dict['state_maxes'], device=device)
        self.MINS = torch.tensor(arg_dict['state_mins'], device=device)
        
        self.PREV_VALS = arg_dict['pv']
        self.actor_loss = []
        self.critic_loss = []
        self.critic_L1loss = []
        self.critic_LNloss = []
            
        if 'HER' in arg_dict['model']:
            self.USE_HER = True
        else:
            self.USE_HER = False

        self.writer = SummaryWriter(arg_dict['tname'])

        self.DISCOUNT = arg_dict['discount']
        self.TAU = arg_dict['tau']
        self.NETWORK_REPL_FREQ = 2
        self.total_it = 0
        self.LOOKBACK_SIZE = 4 # TODO, make this a hyperparameter in the config file

        # Most recent evaluation reward produced by the policy within training
        self.avg_evaluation_reward = 0

        self.BATCH_SIZE = arg_dict['batch_size']
        self.rng = default_rng()
        self.ROLLOUT = self.ROLLOUT_SIZE > 1
        self.u_count = 0
        
        self.actor_component = 100000
        self.critic_component = 10
        self.state_list = arg_dict['state_list']
        self.sampled_positions = np.zeros((100,100))
        self.position_xlims = np.linspace(-0.1, 0.1,100)
        self.position_ylims = np.linspace(0.06,0.26,100)
        self.sampled_file = arg_dict['save_path'] + 'sampled_positions.pkl'
        
        
        try:
            self.CONTACT_SCALING = arg_dict['contact_scaling']
            self.DISTANCE_SCALING = arg_dict['distance_scaling']
        except KeyError:
            self.CONTACT_SCALING = 0.2
            self.DISTANCE_SCALING = 1
        
        print('SCALING SHITE',self.DISTANCE_SCALING, self.CONTACT_SCALING)
        try:
            self.SUCCESS_THRESHOLD = arg_dict['sr']/1000
        except KeyError:
            self.SUCCESS_THRESHOLD = 0.002

    def select_action(self, state: State):
        """
        Method takes in a State object
        Runs state through actor network to get action from policy in numpy array

        :param state: :func:`~mojograsp.simcore.state.State` object.
        :type state: :func:`~mojograsp.simcore.state.State`
        """
        lstate = self.build_state(state)
        lstate = torch.FloatTensor(np.reshape(lstate, (1, -1))).to(device)
        # print('select action call')
        lstate = simple_normalize(lstate, self.MINS, self.MAXES)
        action = self.actor(lstate).cpu().data.numpy().flatten()
        return action

    def grade_action(self, state: State, action: np.ndarray):
        """
        Method takes in a State object and numpy array containing a policy action
        Runs state and action through critic network to return state-action
        q-value (as float) and gradient of q-value relative to the action (as numpy array)

        :param state: :func:`~mojograsp.simcore.state.State` object.
        :param action: :func:`~np.ndarray` containing action
        :type state: :func:`~mojograsp.simcore.state.State`
        :type action: :func:`~np.ndarray` 
        """
        lstate = self.build_state(state)
        lstate = torch.FloatTensor(np.reshape(lstate, (1, -1))).to(device)
        lstate = simple_normalize(lstate, self.MINS, self.MAXES)
        action = torch.tensor(np.reshape(action, (1,-1)), dtype=float, requires_grad=True, device=device)
        action=action.float()
        action.retain_grad()
        g = self.critic(lstate, action)
        g.backward()
        grade = g.cpu().data.numpy().flatten()
        
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
        our_dir = vars(self)
        for key,value in vars(policy_to_copy_from).items():
            if key.isupper():
                our_dir[key] = value
        # self.DISCOUNT = policy_to_copy_from.DISCOUNT
        # self.TAU = policy_to_copy_from.TAU
        # self.ROLLOUT_SIZE = policy_to_copy_from.ROLLOUT_SIZE
        # self.NETWORK_REPL_FREQ = policy_to_copy_from.NETWORK_REPL_FREQ
        self.total_it = policy_to_copy_from.total_it
        self.avg_evaluation_reward = policy_to_copy_from.avg_evaluation_reward

        # Sample from the expert replay buffer, decaying the proportion expert-agent experience over time
        self.sampling_decay_rate = policy_to_copy_from.sampling_decay_rate
        self.sampling_decay_freq = policy_to_copy_from.sampling_decay_freq
        # self.BATCH_SIZE = policy_to_copy_from.BATCH_SIZE

    def save(self, filename):
        """ Save current policy to given filename
		filename: filename to save policy to
        """
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_target.state_dict(), filename + "_critic_target")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_target.state_dict(), filename + "_actor_target")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")
        np.save(filename + "avg_evaluation_reward", np.array([self.avg_evaluation_reward]))

    def save_sampling(self):
        with open(self.sampled_file, 'wb') as f_obj:
            # print(self.sampled_positions)
            pkl.dump(self.sampled_positions, f_obj)
            
    def load(self, filename):
        """ Load input policy from given filename
		filename: filename to load policy from
        """
        self.critic.load_state_dict(torch.load(filename + "_critic", map_location=device))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer", map_location=device))
        self.actor.load_state_dict(torch.load(filename + "_actor", map_location=device))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer", map_location=device))
        self.critic_target.load_state_dict(torch.load(filename + "_critic_target", map_location=device)) 
        self.actor_target.load_state_dict(torch.load(filename + "_actor_target", map_location=device)) 

        # self.critic_target = copy.deepcopy(self.critic)
        # self.actor_target = copy.deepcopy(self.actor)

    def update_target(self):
        """ Update frozen target networks to be closer to current networks
        """
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)

    def build_state(self, state_container: State):
        """
        Method takes in a State object 
        Extracts state information from state_container and returns it as a list based on
        current used states contained in self.state_list

        :param state: :func:`~mojograsp.simcore.phase.State` object.
        :type state: :func:`~mojograsp.simcore.phase.State`
        """
        state = []
        if self.PREV_VALS > 0:
            for i in range(self.PREV_VALS):
                for key in self.state_list:
                    if key == 'op':
                        state.extend(state_container['previous_state'][i]['obj_2']['pose'][0][0:2])
                    elif key == 'ftp':
                        state.extend(state_container['previous_state'][i]['f1_pos'][0:2])
                        state.extend(state_container['previous_state'][i]['f2_pos'][0:2])
                    elif key == 'fbp':
                        state.extend(state_container['previous_state'][i]['f1_base'][0:2])
                        state.extend(state_container['previous_state'][i]['f2_base'][0:2])
                    elif key == 'fcp':
                        state.extend(state_container['previous_state'][i]['f1_contact_pos'][0:2])
                        state.extend(state_container['previous_state'][i]['f2_contact_pos'][0:2])
                    elif key == 'ja':
                        state.extend([item for item in state_container['previous_state'][i]['two_finger_gripper']['joint_angles'].values()])
                    elif key == 'fta':
                        state.extend([state_container['previous_state'][i]['f1_ang'],state_container['previous_state'][i]['f2_ang']])
                    elif key == 'gp':
                        state.extend(state_container['previous_state'][i]['goal_pose']['goal_pose'])
                    else:
                        raise Exception('key does not match list of known keys')

        for key in self.state_list:
            if key == 'op':
                state.extend(state_container['obj_2']['pose'][0][0:2])
            elif key == 'ftp':
                state.extend(state_container['f1_pos'][0:2])
                state.extend(state_container['f2_pos'][0:2])
            elif key == 'fbp':
                state.extend(state_container['f1_base'][0:2])
                state.extend(state_container['f2_base'][0:2])
            elif key == 'fcp':
                state.extend(state_container['f1_contact_pos'][0:2])
                state.extend(state_container['f2_contact_pos'][0:2])
            elif key == 'ja':
                state.extend([item for item in state_container['two_finger_gripper']['joint_angles'].values()])
            elif key == 'fta':
                state.extend([state_container['f1_ang'],state_container['f2_ang']])
            elif key == 'gp':
                state.extend(state_container['goal_pose']['goal_pose'])
            else:
                raise Exception('key does not match list of known keys')
        return state

    def build_reward(self, reward_container: Reward):
        """
        Method takes in a Reward object
        Extracts reward information from state_container and returns it as a float
        based on the reward structure contained in self.REWARD_TYPE

        :param state: :func:`~mojograsp.simcore.reward.Reward` object.
        :type state: :func:`~mojograsp.simcore.reward.Reward`
        """
        if self.REWARD_TYPE == 'Sparse':
            tstep_reward = -1 + 2*(reward_container['distance_to_goal'] < self.SUCCESS_THRESHOLD)
        elif self.REWARD_TYPE == 'Distance':
            tstep_reward = max(-reward_container['distance_to_goal'],-1)
        elif self.REWARD_TYPE == 'Distance + Finger':
            tstep_reward = max(-reward_container['distance_to_goal']*self.DISTANCE_SCALING - max(reward_container['f1_dist'],reward_container['f2_dist'])*self.CONTACT_SCALING,-1)
        elif self.REWARD_TYPE == 'Hinge Distance + Finger':
            tstep_reward = reward_container['distance_to_goal'] < self.SUCCESS_THRESHOLD + max(-reward_container['distance_to_goal'] - max(reward_container['f1_dist'],reward_container['f2_dist'])*self.CONTACT_SCALING,-1)
        elif self.REWARD_TYPE == 'Slope':
            tstep_reward = reward_container['slope_to_goal'] * self.DISTANCE_SCALING
        elif self.REWARD_TYPE == 'Slope + Finger':
            tstep_reward = max(reward_container['slope_to_goal'] * self.DISTANCE_SCALING  - max(reward_container['f1_dist'],reward_container['f2_dist'])*self.CONTACT_SCALING,-1)
        else:
            raise Exception('reward type does not match list of known reward types')
        return float(tstep_reward)
    
    def collect_batch_multistep(self, replay_buffer: ReplayBufferPriority):
        """
        Method takes in a ReplayBufferPriority object
        Extracts BATCH_SIZE transitions and associated priority information from 
        replay buffer and returns them as tensors for training 

        :param state: :func:`~mojograsp.simcore.priority_replay_buffer.ReplayBufferPriority' object.
        :type state: :func:`~mojograsp.simcore.priority_replay_buffer.ReplayBufferPriority
        :return state:  state information as tensor
        :return action: action information as tensor
        :return next_state: next_state information as tensor
        :return reward: reward information as tensor
        :return rollout_reward: rollout reward as tensor
        :return rollout_discount: discount factor to be applied to last state after rollout reward as tensor
        :return last_state: resulting state after rollout as tensor
        :return trimmed_weight: priority of sampled transitions
        :return trimmed_idxs: index in priority queue of sampled transitions
        :return expert_status: bool containing if associated transition is from expert sample
        The name of the next phase or None
        :rtype: str or None
        """
        
        num_timesteps = len(replay_buffer)
        if num_timesteps < self.BATCH_SIZE * 20:
            return None, None, None, None, None, None, None, None, None, None
        else:
            sampled_data, transition_weight, indxs, lookback = replay_buffer.sample_rollout(self.BATCH_SIZE, self.ROLLOUT_SIZE, self.LOOKBACK_SIZE) 
            
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
                    state.append(self.build_state(t_state))
                    action.append(list(timestep[1]['actor_output']))
                    reward.append(self.build_reward(timestep[2]))
                    t_next_state = timestep[3]

                    next_state.append(self.build_state(t_next_state))
                    expert_status.append(timestep[-1])
                    if self.ROLLOUT:
                        j =0
                        for j, timestep in enumerate(timestep_series[1:]):
                            rtemp += self.build_reward(timestep[2]) * self.DISCOUNT ** (j+1)
                        t_last_state = timestep_series[-1][0]
                        rollout_discount.append(j+1)
                        last_state.append(self.build_state(t_last_state))
                        rollout_reward.append(rtemp)
                else:
                    print('timestep series has no length')
                    print(timestep_series, transition_weight, indxs)
            state = torch.tensor(state)
            action = torch.tensor(action)
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
            trimmed_weight = []
            trimmed_idxs = []
            for tw, inds in zip(transition_weight, indxs):
                if len(tw) > 0:
                    trimmed_weight.append(tw[0]) 
                    trimmed_idxs.append(inds[0])
            trimmed_weight = torch.tensor(trimmed_weight)
            trimmed_weight = torch.unsqueeze(trimmed_weight, 1)
            trimmed_weight = trimmed_weight.to(device)
            return state, action, next_state, reward, rollout_reward, rollout_discount, last_state, trimmed_weight, trimmed_idxs, expert_status
        
    def collect_batch(self, replay_buffer: ReplayBufferPriority):
        """
        Method takes in a ReplayBufferPriority object
        Extracts BATCH_SIZE transitions and associated priority information from 
        replay buffer and returns them as tensors for training 

        :param state: :func:`~mojograsp.simcore.priority_replay_buffer.ReplayBufferPriority' object.
        :type state: :func:`~mojograsp.simcore.priority_replay_buffer.ReplayBufferPriority
        :return state:  state information as tensor
        :return action: action information as tensor
        :return next_state: next_state information as tensor
        :return reward: reward information as tensor
        :return rollout_reward: rollout reward as tensor
        :return rollout_discount: discount factor to be applied to last state after rollout reward as tensor
        :return last_state: resulting state after rollout as tensor
        :return trimmed_weight: priority of sampled transitions
        :return trimmed_idxs: index in priority queue of sampled transitions
        :return expert_status: bool containing if associated transition is from expert sample
        The name of the next phase or None
        :rtype: str or None
        """
        
        num_timesteps = len(replay_buffer)
        if num_timesteps < self.BATCH_SIZE * 20:
            return None, None, None, None, None, None, None, None, None, None
        else:
            sampled_data, transition_weight, indxs = replay_buffer.sample_rollout(self.BATCH_SIZE, self.ROLLOUT_SIZE)

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
                    # print(timestep_series)
                    timestep = timestep_series[0]
                    t_state = timestep[0]
                    state.append(self.build_state(t_state))
                    action.append(list(timestep[1]['actor_output']))
                    rtemp = self.build_reward(timestep[2])
                    reward.append(rtemp)
                    t_next_state = timestep[3]

                    next_state.append(self.build_state(t_next_state))
                    expert_status.append(timestep[5])
                    try:
                        x = np.where(t_state['obj_2']['pose'][0][0] < self.position_xlims)[0][0]
                        y = np.where(t_state['obj_2']['pose'][0][1] < self.position_ylims)[0][0]
                        self.sampled_positions[x,y] +=1
                    except IndexError:
                        pass
                    if self.ROLLOUT:
                        series_len = len(timestep_series)
                        for j, timestep in enumerate(timestep_series[1:]):
                            rtemp += self.build_reward(timestep[2]) * self.DISCOUNT ** (j+1)
                        t_last_state = timestep_series[-1][3]
                        rollout_discount.append(series_len)
                        last_state.append(self.build_state(t_last_state))
                        rollout_reward.append(rtemp)
                else:
                    print(timestep_series, transition_weight, indxs)
                    
            state = torch.tensor(state, device=device)
            action = torch.tensor(action, device=device)
            action = action.float()
            reward = torch.tensor(reward, device=device)
            reward = torch.unsqueeze(reward, 1)
            next_state = torch.tensor(next_state, device=device)
            if self.ROLLOUT:
                rollout_reward = torch.tensor(rollout_reward, device=device)
                rollout_reward = torch.unsqueeze(rollout_reward, 1)
                rollout_discount = torch.tensor(rollout_discount, device=device)
                rollout_discount = torch.unsqueeze(rollout_discount, 1)
                last_state = torch.tensor(last_state, device=device)
            else:
                rollout_reward = None
                rollout_discount = None
                last_state = None
            expert_status = torch.tensor(expert_status, device=device)
            expert_status = torch.unsqueeze(expert_status, 1)
            
            
            trimmed_weight = []
            trimmed_idxs = []
            for tw, inds in zip(transition_weight, indxs):
                if len(tw) > 0:
                    trimmed_weight.append(tw[0]) 
                    trimmed_idxs.append(inds[0])
            trimmed_weight = torch.tensor(trimmed_weight, device=device)
            trimmed_weight = torch.unsqueeze(trimmed_weight, 1)
            return state, action, next_state, reward, rollout_reward, rollout_discount, last_state, trimmed_weight, trimmed_idxs, expert_status

    def train(self, replay_buffer):
        """ Update policy based on sample of timesteps from replay buffer"""
        self.total_it += 1

        state, action, next_state, reward, sum_rewards, num_rewards, last_state, transition_weight, indxs, expert_status = self.collect_batch(replay_buffer)
        # print('collect batch just called')
        if state is not None:
            if self.STATE_NOISE > 0:
                state = simple_normalize_noise(state, self.MINS, self.MAXES, self.STATE_NOISE)
                next_state = simple_normalize_noise(next_state, self.MINS, self.MAXES, self.STATE_NOISE)
                if self.ROLLOUT:
                    last_state = simple_normalize_noise(last_state, self.MINS, self.MAXES, self.STATE_NOISE)
            else:
                state = simple_normalize(state, self.MINS, self.MAXES)
                next_state = simple_normalize(next_state, self.MINS, self.MAXES)
                if self.ROLLOUT:
                    last_state = simple_normalize(last_state, self.MINS, self.MAXES)

            
            next_state_val = self.critic_target(next_state, self.actor_target(next_state))

            target_Q = (reward + (self.DISCOUNT * next_state_val).detach()).float()  # bellman equation



            

            # print(target_Q == target_QN)
            # Get current Q estimate
            current_Q = self.critic(state, action)


            scaled_Q = current_Q * transition_weight
            
            scaled_target = target_Q * transition_weight

            
            # L_1 loss (Loss between current state, action and reward, next state, action)
            critic_L1loss = F.mse_loss(scaled_Q, scaled_target)
            # print(self.ROLLOUT)
            if self.ROLLOUT:
                
                # Compute the roll rewards and the number of steps forward (could be less than rollout size if timestep near end of trial)
                super_next_state_val = self.critic_target(last_state, self.actor_target(last_state))
        
                # Compute QN from roll reward and discounted final state
                target_QN = ((sum_rewards.to(device)/num_rewards + (self.DISCOUNT**num_rewards * super_next_state_val).detach())).float()
                # L_2 loss (Loss between current state, action and reward, n state, n action)
                
                            
                scaled_QN = target_QN * transition_weight
                
                critic_LNloss = F.mse_loss(scaled_Q, scaled_QN)


                # Total critic loss
                critic_loss = critic_L1loss.float() + self.LAMBDA_1 * critic_LNloss
            else:
                critic_loss = critic_L1loss.float() 
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_value_(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()

            # Compute actor loss
            actor_action = self.actor(state)
            actor_action.retain_grad()
            individual_actor_loss = -self.critic(state, actor_action)

            actor_loss = individual_actor_loss.mean()
            
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            
            if self.SAMPLING_STRATEGY == 'random+expert':
                priorities = expert_status*0.5 + 0.5
                priorities = priorities.cpu().detach().numpy()
                replay_buffer.update_priorities(indxs,priorities)
                self.writer.add_scalar('priorities/average',np.average(priorities),self.total_it)
            elif self.SAMPLING_STRATEGY == 'random':
                pass
            elif self.SAMPLING_STRATEGY == 'priority':
                actor_component = actor_action.grad.mean(1,True)
                cpart = (current_Q-target_Q)**2
                apart = actor_component**2
                priorities = expert_status*0.5 + 0.0001 + self.actor_component*apart + self.critic_component*cpart
                priorities = priorities.cpu().detach().numpy()
                replay_buffer.update_priorities(indxs,priorities)
                self.writer.add_scalar('priorities/sum_priorities', replay_buffer.buffer_prio.sum(0, replay_buffer.sz),self.total_it)
                # self.writer.add_scalar('priorities/replay_buffer_size', replay_buffer.sz,self.total_it)
                # self.writer.add_scalar('priorities/weights', np.average(transition_weight.cpu().detach().numpy()),self.total_it)
                self.writer.add_scalar('priorities/average', np.average(priorities),self.total_it)
                self.writer.add_scalar('priorities/critic_portion', self.critic_component*np.average(cpart.cpu().detach().numpy()),self.total_it)
                self.writer.add_scalar('priorities/actor_portion', self.actor_component*np.average(apart.cpu().detach().numpy()),self.total_it)
            nn.utils.clip_grad_value_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()

            self.writer.add_scalar('Loss/critic',critic_loss.detach(),self.total_it)
            self.writer.add_scalar('Loss/critic_L1',critic_L1loss.detach(),self.total_it)
            variance = ExplainedVariance()
            ev = variance(current_Q,target_Q)
            self.writer.add_scalar('explained_variance',ev, self.total_it)
            if self.ROLLOUT:
                self.writer.add_scalar('Loss/critic_LN',critic_LNloss.detach(),self.total_it)
            self.writer.add_scalar('Loss/actor',actor_loss.detach(),self.total_it)

            
            # update target networks
            # if self.total_it % self.NETWORK_REPL_FREQ == 0:
            self.update_target()
            self.u_count +=1
            # if self.total_it % 100 ==0:
            #     self.save_poses()
            if self.ROLLOUT:    
                return actor_loss.item(), critic_loss.item(), critic_L1loss.item(), critic_LNloss.item()
            else:
                return actor_loss.item(), critic_loss.item(), critic_L1loss.item(), None

