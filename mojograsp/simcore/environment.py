from abc import ABC, abstractmethod

from mojograsp.simobjects.object_base import ObjectBase

import pybullet as p
import time


class Environment(ABC):
    """Environment Abstract Base Class"""
    @abstractmethod
    def __init__(self):
        """
        Constructor should be used to pass any other objects or 
        data that the environment may need to reset or change.
        Usually this will be a manipulator or other mojograsp objects.
        """
        pass

    @abstractmethod
    def setup(self):
        """
        This method is called by :func:`~mojograsp.simcore.sim_manager.SimManager` once before an episode begins.
        """
        pass

    @abstractmethod
    def step(self):
        """
        This method is called by :func:`~mojograsp.simcore.sim_manager.SimManager` to step pybullet,
        currently the default is a step every 1/240 seconds. Changing step size, length, etc. should
        be done here.
        """
        p.stepSimulation()
        time.sleep(1./240.)

    @abstractmethod
    def reset(self):
        """
        This method is called by :func:`~mojograsp.simcore.sim_manager.SimManager` before every 
        episode. It should be used to reset any sim objects before a new episode begins and optionally
        reload and reset pybullet before each episode.
        """
        pass


class EnvironmentDefault(Environment):
    def __init__(self):
        """Default Placeholder if no Environment class is provided"""
        super().__init__()

    def setup(self):
        """Default Placeholder if no Environment class is provided"""
        super().setup()

    def step(self):
        """Calls default step method in parent :func:`~mojograsp.simcore.environment.Environment`."""
        super().step()

    def reset(self):
        """Default Placeholder if no Environment class is provided"""
        super().reset()
