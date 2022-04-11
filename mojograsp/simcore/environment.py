from abc import ABC, abstractmethod

from mojograsp.simobjects.object_base import ObjectBase

import pybullet as p
import time


class Environment(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def step(self):
        p.stepSimulation()
        time.sleep(1./240.)

    @abstractmethod
    def reset(self):
        pass


class EnvironmentDefault(Environment):
    def __init__(self):
        super().__init__()

    def setup(self):
        super().setup()

    def step(self):
        super().step()

    def reset(self):
        super().reset()
