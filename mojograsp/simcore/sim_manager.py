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
from mojograsp.simcore.phasemanager import PhaseManager
# python imports
import time
import os
# alsdkf
from abc import ABC, abstractmethod
import logging


class SimManager(ABC):
    @abstractmethod
    def add_phase(self, phase_name: str, phase_object: Phase, start: bool = False):
        pass

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def stall(self):
        while p.isConnected():
            time.sleep(1)


class SimManagerDefault(SimManager):

    def __init__(self, num_episodes: int = 1, sim_timestep: float = (1. / 240.), env: Environment = EnvironmentDefault(),
                 episode: Episode = EpisodeDefault(), record: RecordData = RecordDataDefault()):
        self.num_episodes = num_episodes
        self.sim_timestep = sim_timestep
        self.env = env
        self.episode = episode
        self.record = record
        self.phase_manager = PhaseManager()

    def add_phase(self, phase_name: str, phase: Phase, start: bool = False):
        self.phase_manager.add_phase(phase_name, phase, start)

    def run(self):
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
