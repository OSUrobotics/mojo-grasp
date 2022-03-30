from abc import ABC, abstractmethod


class Reward(ABC):
    @abstractmethod
    def get_reward(self) -> dict:
        pass


class RewardBlank(Reward):
    def get_reward(self) -> dict:
        return None
