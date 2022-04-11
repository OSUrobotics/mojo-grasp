
from abc import ABC, abstractmethod
from copy import deepcopy

import json
import logging

from mojograsp.simcore.state import State
from mojograsp.simcore.reward import Reward


class RecordData(ABC):
    @abstractmethod
    def record_timestep(self):
        pass

    @abstractmethod
    def record_episode(self):
        pass

    @abstractmethod
    def save_episode(self):
        pass

    @abstractmethod
    def save_all(self):
        pass


class RecordDataDefault(RecordData):
    def record_timestep(self):
        super().record_timestep()

    def record_episode(self):
        super().record_episode()

    def save_episode(self):
        super().save_episode()

    def save_all(self):
        super().save_all()


class RecordDataJSON(RecordData):

    def __init__(self, data_path: str = None, data_prefix: str = "episode", save_all=False, save_episode=True,
                 state: State = None, reward: Reward = None):
        if not data_path:
            logging.warn("No data path provided")
        self.data_path = data_path
        self.data_prefix = data_prefix
        self.save_all_flag = save_all
        self.save_episode_flag = save_episode
        self.state = state
        self.reward = reward
        self.timestep_num = 1
        self.episode_num = 0
        self.timesteps = []
        self.current_episode = None
        self.episodes = {}

    def record_timestep(self):
        state_reward_dict = {}
        if self.state:
            state_reward_dict["state"] = self.state.get_state()
        if self.reward:
            state_reward_dict["reward"] = self.reward.get_reward()
        timestep_name = "timestep " + str(self.timestep_num)
        timestep_dict = {timestep_name: state_reward_dict}
        self.timesteps.append(timestep_dict)
        self.timestep_num += 1

    def record_episode(self):
        episode_num = "episode_" + str(self.episode_num+1)
        episode = {episode_num: self.timesteps}
        self.current_episode = episode

        if self.save_all_flag:
            self.episodes[episode_num] = self.timesteps

        self.timesteps = []
        self.timestep_num = 1
        self.episode_num += 1

    def save_episode(self):
        if self.save_episode_flag and self.data_path != None:
            file_path = self.data_path + \
                self.data_prefix + "_" + str(self.episode_num)
            with open(file_path, 'w') as fout:
                json.dump(self.current_episode, fout)
        self.current_episode = {}

    def save_all(self):
        if self.save_all_flag and self.data_path != None:
            file_path = self.data_path + \
                self.data_prefix + "_all"
            with open(file_path, 'w') as fout:
                json.dump(self.episodes, fout)
