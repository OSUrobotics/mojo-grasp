
from abc import ABC, abstractmethod
from copy import deepcopy

import json
import logging

from mojograsp.simcore.state import State, StateDefault
from mojograsp.simcore.action import Action, ActionDefault
from mojograsp.simcore.reward import Reward, RewardDefault


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
                 state: State = StateDefault(), action: Action = ActionDefault(), reward: Reward = RewardDefault()):
        if not data_path:
            logging.warn("No data path provided")
        self.data_path = data_path
        self.data_prefix = data_prefix
        self.save_all_flag = save_all
        self.save_episode_flag = save_episode
        self.state = state
        self.action = action
        self.reward = reward
        self.timestep_num = 1
        self.episode_num = 0
        self.timesteps = []
        self.episode_data = []
        self.current_episode = None
        self.episodes = {}

    def record_timestep(self):
        state_reward_dict = {}
        timestep_dict = {"number": self.timestep_num}
        if self.state:
            timestep_dict["state"] = self.state.get_state()
        if self.reward:
            timestep_dict["reward"] = self.reward.get_reward()
        if self.action:
            timestep_dict["action"] = self.action.get_action()
        self.timesteps.append(timestep_dict)
        self.timestep_num += 1

    def record_episode(self):
        episode = {"number": self.episode_num+1}
        episode["timestep_list"] = self.timesteps
        self.current_episode = episode

        if self.save_all_flag:
            self.episode_data.append(episode)

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
                self.episodes = {"episode_list": self.episode_data}
                json.dump(self.episodes, fout)
