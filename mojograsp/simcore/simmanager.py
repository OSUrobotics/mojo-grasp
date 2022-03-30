# pybullet imports
import pybullet as p
import pybullet_data
# mojograsp module imports
from mojograsp.simcore.environment import Environment, EnvironmentDefault
from mojograsp.simcore.phase import Phase
from mojograsp.simcore.state import State
from mojograsp.simcore.episode import Episode, EpisodeBlank
from mojograsp.simcore.record_data import RecordData
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
    def setup(self):
        pass

    @abstractmethod
    def add_phase(self, phase_name: str, phase_object: Phase, start: bool = False):
        pass

    @abstractmethod
    def inject_env(self):
        pass

    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def run(self):
        pass

    def stall(self):
        while p.isConnected():
            time.sleep(1)


class SimManagerDefault(SimManager):

    def __init__(self, num_episodes: int = 1, sim_timestep: float = (1. / 240.), data_path: str = None, gui: bool = True,
                 env: Environment = None, state: State = None, episode: Episode = EpisodeBlank(),
                 record: RecordData = None, replay_buffer: ReplayBuffer = None):
        self.env = env
        self.episode = episode
        self.phase_manager = PhaseManager()

    def setup(self):
        '''Physics server setup.'''
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        self.plane_id = p.loadURDF("plane.urdf")

    def add_phase(self, phase_name: str, phase_object: Phase, start: bool = False):
        self.phase_manager.add_phase(phase_name, phase_object, start)

    def inject_env(self):
        if self.env:
            pass
        pass

    def step(self):
        pass

    def run(self):
        logging.info("RUNNING PHASES: {}".format(
            self.phase_manager.phase_dict))

        for i in range(self.num_episodes):
            self.episode.setup()
            self.phase_manager.exit_flag = False
            self.phase_manager.start_phases()
            current_phase = self.phase_manager.current_phase

            while self.phase_manager.exit_flag == False:
                current_phase.setup()
                done = False
                logging.info("CURRENT PHASE: {}".format(
                    self.phase_manager.current_phase.name))
                while not done:
                    current_phase.pre_step()
                    self.step()
                    current_phase.post_step()
                    done = current_phase.exit_condition()
                self.phase_manager.get_next_phase()

            self.episode.post_episode()
            self.episode.reset()

        logging.info("COMPLETED")
