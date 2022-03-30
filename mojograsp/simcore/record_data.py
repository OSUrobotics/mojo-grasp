from abc import ABC, abstractmethod
from copy import deepcopy


class RecordData(ABC):
    sim = None

    @abstractmethod
    def record_timestep(self):
        pass

    @abstractmethod
    def record_episode(self):
        pass

    @abstractmethod
    def save_episode(self):
        pass

    def save_all(self):
        pass


class RecordDataDefault(RecordData):
    def __init__(self, data_path: str = None, data_prefix: str = "Episode", save_all=False, save_episode=True):
        self.data_path = data_path
        self.data_prefix = data_prefix
        self.save_all_flag = save_all
        self.save_episode_flag = save_episode
        self.timestep_num = 0

        self.timesteps = []
        self.episodes = []

        self.state = None
        self.reward = None
        if self.sim:
            self.state = self.sim.state
            self.reward = self.sim.reward

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
        print(self.episodes[-1])

    def save_all(self):
        pass
