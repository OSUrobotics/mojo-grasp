from abc import ABC, abstractmethod


class Reward(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_reward(self) -> dict:
        pass


class RewardDefault(Reward):
    def __init__(self):
        super().__init__()

    def get_reward(self) -> dict:
        super().get_reward()
