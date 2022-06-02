from collections import deque
from dataclasses import dataclass, asdict
import dataclasses
from abc import ABC, abstractmethod
import json
import logging
from mojograsp.simcore.action import Action, ActionDefault
from mojograsp.simcore.reward import Reward, RewardDefault
from mojograsp.simcore.state import State, StateDefault


@dataclass
class Timestep:
    episode: int = None
    timestep: int = None
    state: dict = None
    action: dict = None
    reward: dict = None
    next_state: dict = None


class ReplayBuffer(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def add_timestep(self, episode_num: int, timestep_num: int):
        pass

    @abstractmethod
    def save_buffer(self, filename: str):
        pass


class ReplayBufferDefault:
    def __init__(self, buffer_size: int = 10000, no_delete=False, state: State = StateDefault,
                 action: Action = ActionDefault, reward: Reward = RewardDefault):
        '''Initializes episode file being used and the replay buffer structure'''
        self.buffer_size = buffer_size
        self.prev_timestep = None
        self.state = state
        self.reward = reward
        self.action = action

        # deque data structure deletes oldest entry in array once buffer_size is exceeded
        if no_delete:
            self.buffer = deque()
        else:
            self.buffer = deque(maxlen=buffer_size)

    def load_buffer_JSON(self, file_path: str):
        self.buffer.clear()
        with open(file_path) as f:
            data = json.load(f)
            for i in data["episode_list"]:
                for j in i["timestep_list"]:
                    tstep = Timestep(episode=int(i["number"]), timestep=int(
                        j["number"]), state=j["state"], action=j["action"], reward=j["reward"])
                    self.backfill(tstep)
                self.prev_timestep = None

    def load_buffer_BUFFER(self, file_path: str):
        self.buffer.clear()
        with open(file_path) as f:
            data = json.load(f)
            for i in data:
                tstep = Timestep(episode=int(i["episode"]), timestep=int(
                    i["timestep"]), state=i["state"], action=i["action"], reward=i["reward"],
                    next_state=i["next_state"])
                self.buffer.append(tstep)
            self.prev_timestep = None

    def backfill(self, tstep: Timestep):
        if self.prev_timestep and self.prev_timestep != tstep.episode:
            self.prev_timestep == None

        if self.prev_timestep:
            self.prev_timestep.next_state = tstep.state
            self.buffer.append(self.prev_timestep)
        self.prev_timestep = tstep

    def add_timestep(self, episode_num: int, timestep_num: int):
        tstep = Timestep(episode=episode_num, timestep=timestep_num, state=self.state.get_state,
                         action=self.action.get_action, reward=self.reward.get_reward)
        self.backfill(tstep)

    def save_buffer(self, file_path: str = None):
        with open(file_path, 'w') as fout:
            temp_list = list(self.buffer)
            temp_list = [asdict(x) for x in temp_list]
            json.dump(temp_list, fout)
