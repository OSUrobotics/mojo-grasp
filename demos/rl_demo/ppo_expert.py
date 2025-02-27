import warnings
from typing import Any, ClassVar, Optional, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F
import os
import re
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, \
    MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.utils import obs_as_tensor
from collections import OrderedDict
from copy import deepcopy

from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
import time
import sys
from abc import ABC
import pickle as pkl
import multiprocessing

SelfPPO = TypeVar("SelfPPO", bound="PPO")

def simple_build_state(state_container, key_list):
    """
    Method takes in a State object 
    Extracts state information from state_container and returns it as a list based on
    current used states contained in self.state_list

    :param state: :func:`~mojograsp.simcore.phase.State` object.
    :type state: :func:`~mojograsp.simcore.phase.State`
    """
    angle_keys = ["finger0_segment0_joint","finger0_segment1_joint","finger1_segment0_joint","finger1_segment1_joint"]
    state = []
    PREV_VALS = 2
    #print('state list!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', self.state_list)
    if PREV_VALS > 0:
        for i in range(PREV_VALS):
            for key in key_list:
                #Changed This to x,y,z from x,y
                if key == 'op':
                    state.extend(state_container['previous_state'][i]['obj_2']['pose'][0][0:3])
                elif key == 'oo':
                    state.extend(state_container['previous_state'][i]['obj_2']['pose'][1])
                elif key == 'oa':
                    state.extend([np.sin(state_container['previous_state'][i]['obj_2']['z_angle']),np.cos(state_container['previous_state'][i]['obj_2']['z_angle'])])
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
                    state.extend([state_container['previous_state'][i]['two_finger_gripper']['joint_angles'][item] for item in angle_keys])
                elif key == 'fta':
                    state.extend([state_container['previous_state'][i]['f1_ang'],state_container['previous_state'][i]['f2_ang']])
                elif key == 'eva':
                    state.extend(state_container['previous_state'][i]['two_finger_gripper']['eigenvalues'])
                elif key == 'evc':
                    state.extend(state_container['previous_state'][i]['two_finger_gripper']['eigenvectors'])
                elif key == 'evv':
                    evecs = state_container['previous_state'][i]['two_finger_gripper']['eigenvectors']
                    evals = state_container['previous_state'][i]['two_finger_gripper']['eigenvalues']
                    scaled = [evals[0]*evecs[0],evals[0]*evecs[2],evals[1]*evecs[1],evals[1]*evecs[3],
                                evals[2]*evecs[4],evals[2]*evecs[6],evals[3]*evecs[5],evals[3]*evecs[7]]
                    state.extend(scaled)
                elif key == 'params':
                    state.extend(state_container['hand_params'])
                elif key == 'gp':
                    state.extend(state_container['previous_state'][i]['goal_pose']['goal_position'])
                elif key == 'go':
                    state.append(state_container['previous_state'][i]['goal_pose']['goal_orientation'])
                elif key == 'gf':
                    state.extend(state_container['previous_state'][i]['goal_pose']['goal_finger'])
                elif key == 'wall':
                    state.extend(state_container['previous_state'][i]['wall']['pose'][0][0:2])
                    state.extend(state_container['previous_state'][i]['wall']['pose'][1][0:4])
                # What Jeremiah Added
                elif key == 'slice':
                    pass
                elif key == 'mslice':

                    state.extend(state_container['previous_state'][i]['dynamic'].flatten())
                elif key == 'latent':
                    state.extend(state_container['previous_state'][i]['latent'])
                elif key == 'rad':
                    state.append(state_container['previous_state'][i]['f1_contact_distance'])
                    state.append(state_container['previous_state'][i]['f2_contact_distance'])
                    state.append(state_container['previous_state'][i]['f1_contact_flag'])
                    state.append(state_container['previous_state'][i]['f2_contact_flag'])
                elif key == 'ra':
                    state.append(state_container['previous_state'][i]['f1_contact_angle'])
                    state.append(state_container['previous_state'][i]['f2_contact_angle'])
                else:
                    raise Exception('key does not match list of known keys')

    for key in key_list:
        # changed to x,y,z from x,y
        if key == 'op':
            state.extend(state_container['obj_2']['pose'][0][0:3])
        elif key == 'oo':
            state.extend(state_container['obj_2']['pose'][1])
        elif key == 'oa':
            state.extend([np.sin(state_container['obj_2']['z_angle']),np.cos(state_container['obj_2']['z_angle'])])
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
            state.extend([state_container['two_finger_gripper']['joint_angles'][item] for item in angle_keys])
        elif key == 'fta':
            state.extend([state_container['f1_ang'],state_container['f2_ang']])
        elif key == 'eva':
            state.extend(state_container['two_finger_gripper']['eigenvalues'])
        elif key == 'evc':
            state.extend(state_container['two_finger_gripper']['eigenvectors'])
        elif key == 'evv':
            evecs = state_container['two_finger_gripper']['eigenvectors']
            evals = state_container['two_finger_gripper']['eigenvalues']
            scaled = [evals[0]*evecs[0],evals[0]*evecs[2],evals[1]*evecs[1],evals[1]*evecs[3],
                        evals[2]*evecs[4],evals[2]*evecs[6],evals[3]*evecs[5],evals[3]*evecs[7]]
            state.extend(scaled)
        elif key == 'params':
            state.extend(state_container['hand_params'])
        elif key == 'gp':
            state.extend(state_container['goal_pose']['goal_position'])
        elif key == 'go':
            state.append(state_container['goal_pose']['goal_orientation'])
        elif key == 'gf':
            state.extend(state_container['goal_pose']['goal_finger'])
        elif key == 'wall':
            state.extend(state_container['wall']['pose'][0][0:2])
            state.extend(state_container['wall']['pose'][1][0:4])
            
        # What Jeremiah Added
        elif key == 'slice':
            # print(state_container)
            state.extend(state_container['slice'].flatten())
        elif key == 'mslice':

            state.extend(state_container['dynamic'].flatten())
        elif key == 'latent':
            state.extend(state_container['latent'])

        elif key == 'rad':
            state.append(state_container['f1_contact_distance'])
            state.append(state_container['f2_contact_distance'])
            state.append(state_container['f1_contact_flag'])
            state.append(state_container['f2_contact_flag'])
        elif key == 'ra':
            state.append(state_container['f1_contact_angle'])
            state.append(state_container['f2_contact_angle'])

        else:
            raise Exception('key does not match list of known keys')
        
    return state

def pool_data_extraction(episode_file, build_state, key_list, reward_function, weights):
    '''
    This takes a given episode file, a build state function and a set of keys and returns a list
    of all the data from that episode processed into data that can be fed directly into the network.
    Designed to be used with Pool.starmap on a folder 
    full of data
    '''
    with open(episode_file, 'rb') as ef:
        tempdata = pkl.load(ef)
    data = tempdata['timestep_list']
    next_state = []
    state = []
    action = []
    reward = []
    done = []
    firsts = []
    start_state = tempdata['start_state']
    for i in range(len(data)):
        data[len(data)-1-i]['state']['previous_state'] = []
        for j in range(4):
            if len(data)-1-i-4+j < 0:
                # print('too soon, just using the first one')
                data[len(data)-1-i]['state']['previous_state'].append(deepcopy(start_state))
            else:
                data[len(data)-1-i]['state']['previous_state'].append(deepcopy(data[len(data)-1-i-4+j]['state']))
        next_state.append(build_state(data[len(data)-1-i]['state'], key_list))
        if i != 0:
            state.append(build_state(data[len(data)-1-i]['state'], key_list))
        reward.append(reward_function(data[len(data)-1-i]['reward'], weights)[0])
        action.append(data[len(data)-1-i]['action']['actor_output'])
        done.append(int((len(data)-1-i)==0))
        firsts.append(int(i==0))
    state_1 = deepcopy(start_state)
    state_1['previous_state'] = []
    for i in range(4):
        state_1['previous_state'].append(deepcopy(start_state))
    state.append(build_state(state_1, key_list))
    state.reverse()
    next_state.reverse()
    action.reverse()
    reward.reverse()
    done.reverse()
    firsts.reverse()
    return {'observation':state,'action':action,'reward':reward,'next_state':next_state,'done':done,'episode_start':firsts}

class PPOExpert(OnPolicyAlgorithm, ABC):
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)

    Paper: https://arxiv.org/abs/1707.06347
    Code: This implementation borrows code from OpenAI Spinning Up (https://github.com/openai/spinningup/)
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    Stable Baselines (PPO2 from https://github.com/hill-a/stable-baselines)

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
        NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
        See https://github.com/pytorch/pytorch/issues/29372
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param normalize_advantage: Whether to normalize or not the advantage
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param rollout_buffer_class: Rollout buffer class to use. If ``None``, it will be automatically selected.
    :param rollout_buffer_kwargs: Keyword arguments to pass to the rollout buffer on creation
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation. See :ref:`ppo_policies`
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def __init__(
            self,
            policy: Union[str, type[ActorCriticPolicy]],
            env: Union[GymEnv, str],
            learning_rate: Union[float, Schedule] = 3e-4,
            n_steps: int = 100,
            batch_size: int = 32,
            n_epochs: int = 10,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            clip_range: Union[float, Schedule] = 0.2,
            clip_range_vf: Union[None, float, Schedule] = None,
            normalize_advantage: bool = True,
            ent_coef: float = 0.0,
            vf_coef: float = 0.5,
            max_grad_norm: float = 0.5,
            use_sde: bool = False,
            sde_sample_freq: int = -1,
            rollout_buffer_class: Optional[type[RolloutBuffer]] = None,
            rollout_buffer_kwargs: Optional[dict[str, Any]] = None,
            target_kl: Optional[float] = None,
            stats_window_size: int = 100,
            tensorboard_log: Optional[str] = None,
            policy_kwargs: Optional[dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        if normalize_advantage:
            assert (
                    batch_size > 1
            ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert buffer_size > 1 or (
                not normalize_advantage
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl

        self.num_expert_envs = self.env.num_envs

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)
        buffer_cls = DictRolloutBuffer if isinstance(self.observation_space, spaces.Dict) else RolloutBuffer

        self.expert_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

    def load_expert_translation(self, folder, build_state, key_list, reward_function, reward_weights, n_rollout_steps) -> bool:
        # Make a list of offline observations, actions and trees
        print("INFO: Making offline rollouts")
        n_steps = 0
        episode_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith('.pkl')]

        episode_files = np.array(episode_files)

        thing = [[ef, build_state, key_list, reward_function, reward_weights] for ef in episode_files]
        print('applying async')
        pool = multiprocessing.Pool()
        data_list = pool.starmap(pool_data_extraction,thing)
        data_dict = {'observation':[],'action':[],'reward':[],'next_state':[],'done':[],'firsts':[]}
        # Sample expert episode
        print('done async')
        for episode_dict in data_list:
            data_dict['observation'].extend(episode_dict['observation'])
            data_dict['action'].extend(episode_dict['action'])
            data_dict['reward'].extend(episode_dict['reward'])
            data_dict['next_state'].extend(episode_dict['next_state'])
            data_dict['done'].extend(episode_dict['done'])
            data_dict['firsts'].extend(episode_dict['episode_start'])
        
        data_dict['observation'] = np.array(data_dict['observation'])
        data_dict['action'] = np.array(data_dict['action'])
        data_dict['reward'] = np.array(data_dict['reward'])
        data_dict['next_state'] = np.array(data_dict['next_state'])
        data_dict['done'] = np.array(data_dict['done'])
        data_dict['firsts'] = np.array(data_dict['firsts'])
        new_inds = np.random.shuffle(list(range(len(data_dict['observation']))))
        data_dict['observation'] = data_dict['observation'][new_inds]
        data_dict['action'] = data_dict['action'][new_inds]
        data_dict['reward'] = data_dict['reward'][new_inds]
        data_dict['next_state'] = data_dict['next_state'][new_inds]
        data_dict['done'] = data_dict['done'][new_inds]
        data_dict['firsts'] = data_dict['firsts'][new_inds]
        # self.expert_buffer = RolloutBuffer(n_rollout_steps,self.observation_space,self.action_space)
        with th.no_grad():
            # Convert to pytorch tensor or to TensorDict
            obs_tensor = obs_as_tensor(data_dict['observation'][0], 'cuda')
            actions = obs_as_tensor(data_dict['action'][0], 'cuda')
            next_state = data_dict['next_state'][0]
            values, log_probs,_ = self.policy.evaluate_actions(obs_tensor, actions)
            # rewards = obs_as_tensor(data_dict['reward'][0],'cuda')
            dones = obs_as_tensor(data_dict['done'][0],'cuda')
            # firsts = obs_as_tensor(data_dict['firsts'][0],'cpu')

        while n_steps < n_rollout_steps:
            # actions = obs_as_tensor(data_dict['reward'][n_steps], 'cuda')
            # print(np.shape(data_dict['observation']), dones,rewards)
            self.expert_buffer.add(
                data_dict['observation'][0][n_steps:n_steps+16],
                data_dict['action'][0][n_steps:n_steps+16],
                data_dict['reward'][0][n_steps:n_steps+16],
                data_dict['firsts'][0][n_steps:n_steps+16],
                values[n_steps:n_steps+16],
                log_probs[n_steps:n_steps+16],
            )

            self._last_episode_starts_expert = dones
            n_steps+=16

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(
                obs_as_tensor(next_state, self.device))  # pylint: disable=unexpected-keyword-arg
        print(type(values),values.size(),type(dones),dones.size())
        self.expert_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        if self.verbose > 0:
            print("INFO: Finished making offline rollouts")
        # callback.on_rollout_end()
        # callback.update_locals(locals())

        return True

    def train_ppo_offline(self, clip_range, clip_range_vf) -> None:
        self.policy.set_training_mode(True)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True
        # Logging dict
        logs_online = {"ratio_current_old": [], "policy_loss": [], "value_loss": [], "entropy_loss": [],
                       "approx_kl_div": [],
                       "clip_fraction": [], "advantages": [], "log_prob": []}

        logs_offline = {"ratio_current_old": [], "policy_loss": [], "value_loss": [], "entropy_loss": [],
                        "approx_kl_div": [],
                        "clip_fraction": [], "clamp_fraction": [], "advantages": [], "max_log_prob": [],
                        "min_log_prob": [], "log_prob": []}
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            offline_data_buffer = self.expert_buffer.get(self.batch_size)
            online_data_buffer = self.rollout_buffer.get(self.batch_size)
            # Do a complete pass on the rollout buffer
            while True:
                try:
                    # Get offline batch from buffer
                    offline_batch = next(offline_data_buffer)
                    # Get online batch from buffer
                    online_batch = next(online_data_buffer)
                except StopIteration:
                    break

                actions_offline = offline_batch.actions
                actions_online = online_batch.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions_online = online_batch.actions.long().flatten()
                    actions_offline = offline_batch.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                # Evaluate online actions
                values_online, log_prob_online, entropy_online = self.policy.evaluate_actions(
                    online_batch.observations,
                    actions_online,
                )

                # Evaluate offline actions
                values_offline, log_prob_offline, entropy_offline, = self.policy.evaluate_actions(
                    offline_batch.observations,
                    actions_offline,
                )

                min_log_prob = -10
                log_prob_offline = th.clamp(log_prob_offline, min_log_prob)
                log_prob_expert = 5  # Dial to weigh expert data vs online data. If expert data is from a similar policy (Inputs/Outputs) keep it low

                values_online = values_online.flatten()
                values_offline = values_offline.flatten()

                # Importance sampling ratios
                ratio_old_expert_offline = th.exp(
                    offline_batch.old_log_prob - log_prob_expert)
                ratio_current_old_online = th.exp(log_prob_online - online_batch.old_log_prob)
                ratio_current_old_offline = th.exp(
                    log_prob_offline - th.clamp(offline_batch.old_log_prob, min_log_prob, 100))
                ratio_current_expert_offline = th.exp(
                    log_prob_offline - log_prob_expert)

                # Normalize advantage
                advantages_online = online_batch.advantages

                advantages_offline = offline_batch.advantages * ratio_old_expert_offline
                # Concatenate advantages
                advantages = th.cat((advantages_online, advantages_offline), 0)
                #Clamp advantages
                advantages = th.clamp(advantages, -0.3, 0.3)
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                advantages_online = advantages[:len(advantages_online)]
                advantages_offline = advantages[len(advantages_online):]

                # clipped surrogate loss for online
                policy_loss_1_online = advantages_online * ratio_current_old_online
                policy_loss_2_online = advantages_online * th.clamp(ratio_current_old_online, 1 - clip_range,
                                                                    1 + clip_range)
                policy_loss_online = -th.mean(th.min(policy_loss_1_online, policy_loss_2_online))

                # clipped surrogate loss for offline
                policy_loss_1_offline = advantages_offline * ratio_current_old_offline  # Old/expert ratio is already multiplied
                # old/expert*current/old = current/expert (Expert sampled the data)
                policy_loss_2_offline = advantages_offline * th.clamp(ratio_current_old_offline, 1 - clip_range,
                                                                      1 + clip_range)
                policy_loss_offline = -th.mean(th.min(policy_loss_1_offline, policy_loss_2_offline))

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred_online = values_online
                    values_pred_offline = values_offline
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred_online = online_batch.old_values + th.clamp(
                        values_online - online_batch.old_values, -clip_range_vf, clip_range_vf
                    )
                    values_pred_offline = offline_batch.old_values + th.clamp(
                        values_offline - offline_batch.old_values, -clip_range_vf, clip_range_vf
                    )

                # Value loss using the TD(gae_lambda) target
                value_loss_online = th.mean(
                    (((online_batch.returns - values_pred_online) ** 2))) * self.vf_coef
                value_loss_offline = th.mean(
                    (((offline_batch.returns - values_pred_offline) ** 2) * ratio_old_expert_offline)) * self.vf_coef

                if entropy_online is None:
                    # Approximate entropy when no analytical form
                    entropy_loss_online = -th.mean(-log_prob_online + 1e-8) * self.ent_coef
                else:
                    entropy_loss_online = -th.mean(entropy_online) * self.ent_coef

                if entropy_offline is None:
                    # Approximate entropy when no analytical form
                    entropy_loss_offline = -th.mean(-log_prob_offline + 1e-8) * self.ent_coef
                else:
                    entropy_loss_offline = -th.mean(entropy_offline) * self.ent_coef

                online_loss = policy_loss_online + entropy_loss_online + value_loss_online + entropy_loss_online
                offline_loss = policy_loss_offline + value_loss_offline/10  # TODO: Regularization here for offline actions with -ve advantage

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    # Approx KL Divergence -- Try to keep them vv low (For online+BC this value below 0.02 works)
                    log_ratio_online = log_prob_online - online_batch.old_log_prob
                    approx_kl_div_online = th.mean(
                        ((th.exp(log_ratio_online) - 1) - log_ratio_online)).cpu().numpy()
                    log_ratio_offline = log_prob_offline - th.clamp(offline_batch.old_log_prob, min_log_prob, 100)
                    approx_kl_div_offline = th.mean(
                        ((th.exp(log_ratio_offline) - 1) - log_ratio_offline)).cpu().numpy()

                    # Adjust learning rate to keep clip at about 0.2 - 0.3
                clip_fraction_online = th.mean(
                    (th.abs(ratio_current_old_online - 1) > clip_range).float()).item()
                clip_fraction_offline = th.mean(
                    (th.abs(ratio_current_old_offline - 1) > clip_range).float()).item()
                clamp_fraction_offline = th.mean((log_prob_offline <= min_log_prob).float()).item()

                # This is all just a logging
                # Log online data
                logs_online["ratio_current_old"].append(ratio_current_old_online.mean().item())
                logs_online["policy_loss"].append(policy_loss_online.item())
                logs_online["value_loss"].append(value_loss_online.item())
                logs_online["entropy_loss"].append(entropy_loss_online.item())
                logs_online["approx_kl_div"].append(approx_kl_div_online)
                logs_online["clip_fraction"].append(clip_fraction_online)
                logs_online["advantages"].append(advantages_online.mean().item())
                logs_online["log_prob"].append(log_prob_online.mean().item())
                logs_online["advantages"].append(advantages_online.mean().item())

                # Log offline data
                logs_offline["ratio_current_old"].append(ratio_current_old_offline.mean().item())
                logs_offline["policy_loss"].append(policy_loss_offline.item())
                logs_offline["value_loss"].append(value_loss_offline.item())
                logs_offline["entropy_loss"].append(entropy_loss_offline.item())
                logs_offline["approx_kl_div"].append(approx_kl_div_offline)
                logs_offline["clip_fraction"].append(clip_fraction_offline)
                logs_offline["clamp_fraction"].append(clamp_fraction_offline)
                logs_offline["advantages"].append(advantages_offline.mean().item())
                logs_offline["max_log_prob"].append(log_prob_offline.max().item())
                logs_offline["min_log_prob"].append(log_prob_offline.min().item())
                logs_offline["log_prob"].append(log_prob_offline.mean().item())
                logs_offline["ratio_old_expert"].append(ratio_old_expert_offline.mean().item())

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss_offline = offline_loss / 2
                loss_offline.backward()
                # # Log std should only be updated for online data
                self.policy.log_std.grad.zero_()
                loss_online = online_loss / 2
                loss_online.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

        self._n_updates += 1

        explained_var_online = explained_variance(self.rollout_buffer.values.flatten(),
                                                  self.rollout_buffer.returns.flatten())
        explained_var_offline = explained_variance(self.expert_buffer.values.flatten(),
                                                   self.expert_buffer.returns.flatten())
        # Logs
        # Log online data
        explained_var_online = explained_variance(self.rollout_buffer.values.flatten(),
                                                  self.rollout_buffer.returns.flatten())
        explained_var_offline = explained_variance(self.expert_buffer.values.flatten(),
                                                   self.expert_buffer.returns.flatten())
        # Logs
        # Log online data
        for key, value in logs_online.items():
            self.logger.record("train_online/{}".format(key), safe_mean(value))
        # Log offline data
        for key, value in logs_offline.items():
            self.logger.record("train_offline/{}".format(key), safe_mean(value))

        self.logger.record("train_online/explained_variance", explained_var_online)
        self.logger.record("train_offline/explained_variance_offline", explained_var_offline)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        self.log_from_rollout_buffer(self.expert_buffer, 'train_offline/')
        self.log_from_rollout_buffer(self.rollout_buffer, 'train_online/')

        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def log_from_rollout_buffer(self, buffer, prefix):
        self.logger.record(prefix + "returns", np.mean(buffer.returns))
        self.logger.record(prefix + "values", np.mean(buffer.values))
        self.logger.record(prefix + "advantages_buffer", np.mean(buffer.advantages))
        self.logger.record(prefix + "explained_variance",
                           explained_variance(buffer.values.flatten(), buffer.returns.flatten()))


    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)

        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]
        self.train_ppo_offline(clip_range, clip_range_vf)

    def learn(
            self: SelfPPO,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 1,
            tb_log_name: str = "PPO",
            reset_num_timesteps: bool = True,
            progress_bar: bool = False,
    ) -> SelfPPO:
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None

        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer,
                                                      n_rollout_steps=self.n_steps)
            continue_training = self.make_offline_rollouts(callback, self.expert_buffer,
                                                           n_rollout_steps=self.n_steps)

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
                fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record("rollout/ep_rew_mean",
                                       safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_len_mean",
                                       safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)

            self.train()

        callback.on_training_end()

        return self
