from abc import ABC, abstractmethod

from mojograsp.simcore.state import State
from mojograsp.simcore.reward import Reward
from mojograsp.simobjects.hand import Hand
from mojograsp.simobjects.objectbase import ObjectBase


class Environment(ABC):
    @abstractmethod
    def __init__(self):
        pass


class EnvironmentDefault(Environment):
    # TODO Needs fleshing out, and safety checks
    def __init__(self, hand: Hand = None, object: ObjectBase = None, state: State = None, reward: Reward = None):
        self.hand = hand
        self.object = object
        self.state = state
        self.reward = reward
