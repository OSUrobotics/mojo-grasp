from abc import ABC, abstractmethod
from copy import deepcopy

from mojograsp.simcore.state import State
from mojograsp.simcore.reward import Reward


class RecordData(ABC):
    @abstractmethod
    def __init__(self):
        pass

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
    def __init__(self):
        super().__init__()

    def record_timestep(self):
        super().record_timestep()

    def record_episode(self):
        super().record_episode()

    def save_episode(self):
        super().save_episode()

    def save_all(self):
        super().save_all()


class RecordDataJSON(RecordData):
    def __init__(self, data_path: str = None, data_prefix: str = "Episode", save_all=False, save_episode=True,
                 state: State = None, reward: Reward = None):
        self.data_path = data_path
        self.data_prefix = data_prefix
        self.save_all_flag = save_all
        self.save_episode_flag = save_episode
        self.timestep_num = 0

        self.timesteps = []
        self.episodes = []

        self.state = state
        self.reward = reward

    def record_timestep(self):
        tstep_state = None
        tstep_reward = None
        state_reward_dict = {"timestep", self.timestep_num}
        if self.state:
            tstep_state = self.state.get_state()
            if tstep_state:
                state_reward_dict.update(tstep_state)
        if self.reward:
            tstep_reward = self.reward.get_reward()
            if tstep_reward:
                state_reward_dict.update(tstep_reward)

        self.timesteps.append(state_reward_dict)
        self.timestep_num += 1

    def record_episode(self):
        self.episodes.append(self.timesteps)
        self.timesteps = []
        self.timestep_num = 0

    def save_episode(self):
        super().save_episode

    def save_all(self):
        super().save_all
