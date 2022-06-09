# pybullet imports
import pybullet as p
import pybullet_data
# mojograsp module imports
from mojograsp.simcore.environment import Environment, EnvironmentDefault
from mojograsp.simcore.phase import Phase
from mojograsp.simcore.reward import Reward
from mojograsp.simcore.state import State
from mojograsp.simcore.episode import Episode, EpisodeDefault
from mojograsp.simcore.record_data import RecordData, RecordDataDefault
from mojograsp.simcore.replay_buffer import ReplayBuffer
from mojograsp.simcore.phase_manager import PhaseManager
# python imports
import time
import os
# alsdkf
from abc import ABC, abstractmethod
import logging


class SimManager(ABC):
    """SimManager Abstract Base Class"""
    @abstractmethod
    def add_phase(self, phase_name: str, phase_object: Phase, start: bool = False):
        """
        Method takes in a phase name, phase object and a boolean to specify whether it is the starting phase.
        Adds the phase to the phase manager

        :param phase_name: Name of the phase being added.
        :param phase: :func:`~mojograsp.simcore.phase.Phase` object.
        :param start: If True then the phase given will be the starting phase.
        :type phase_name: str
        :type phase: :func:`~mojograsp.simcore.phase.Phase` 
        :type start: bool
        """
        pass

    @abstractmethod
    def run(self):
        """Method begins the episode and phase loop until all episodes are completed"""
        pass

    @abstractmethod
    def stall(self):
        """Helper method to keep sim open for inspection and stalled until window is closed"""
        while p.isConnected():
            time.sleep(1)


class SimManagerDefault(SimManager):
    """
    Default SimManager class, runs through all episodes and uses the passed in
    mojograsp classes to call the necessary functions. This default will work for any non RL application.  
    """

    def __init__(self, num_episodes: int = 1, env: Environment = EnvironmentDefault(),
                 episode: Episode = EpisodeDefault(), record: RecordData = RecordDataDefault()):
        """
        Constructor passes in the environment, episode and record data objects and simmanager parameters.

        :param num_episodes: Number of episodes to run.
        :param env: :func:`~mojograsp.simcore.environment.Environment` object
        :param episode: :func:`~mojograsp.simcore.episode.Episode` object
        :param record: :func:`~mojograsp.simcore.record_data.RecordData` object
        :type num_episodes: int
        :type env: :func:`~mojograsp.simcore.environment.Environment`
        :type episode: :func:`~mojograsp.simcore.episode.Episode`
        :type record: :func:`~mojograsp.simcore.record_data.RecordData`
        """
        self.num_episodes = num_episodes
        self.env = env
        self.episode = episode
        self.record = record
        self.phase_manager = PhaseManager()

    def add_phase(self, phase_name: str, phase: Phase, start: bool = False):
        """
        Method takes in a phase name, phase object and a boolean to specify whether it is the starting phase.
        Adds the phase to the phase manager

        :param phase_name: Name of the phase being added.
        :param phase: :func:`~mojograsp.simcore.phase.Phase` object.
        :param start: If True then the phase given will be the starting phase.
        :type phase_name: str
        :type phase: :func:`~mojograsp.simcore.phase.Phase` 
        :type start: bool
        """

        self.phase_manager.add_phase(phase_name, phase, start)

    def run(self):
        """
        Runs through all episodes, and the phases in each episode, continues until all episodes are completed.
        Calls the user defined phases and other classes inherited from the abstract base classes. Detailed diagram
        is above of the order of operations.
        """
        if len(self.phase_manager.phase_dict) > 0:
            logging.info("RUNNING PHASES: {}".format(
                self.phase_manager.phase_dict))
            for i in range(self.num_episodes):
                self.episode.setup()
                self.env.setup()
                self.phase_manager.set_exit_flag(False)
                self.phase_manager.setup()

                while self.phase_manager.get_exit_flag() == False:
                    self.phase_manager.current_phase.setup()
                    done = False
                    logging.info("CURRENT PHASE: {}".format(
                        self.phase_manager.current_phase.name))
                    while not done:
                        self.phase_manager.current_phase.pre_step()
                        self.phase_manager.current_phase.execute_action()
                        self.record.record_timestep()
                        self.env.step()
                        self.phase_manager.current_phase.post_step()
                        done = self.phase_manager.current_phase.exit_condition()
                    self.phase_manager.get_next_phase()

                self.record.record_episode()
                self.record.save_episode()
                self.episode.post_episode()
                self.env.reset()
            self.record.save_all()
        else:
            logging.warn("No Phases have been added")
        print("COMPLETED")

    def stall(self):
        super().stall()
