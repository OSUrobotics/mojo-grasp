# pybullet imports
from mojograsp.simcore.state import StateDefault
import pybullet as p
# mojograsp module imports
from mojograsp.simcore.environment import Environment, EnvironmentDefault
from mojograsp.simcore.phase import Phase
from mojograsp.simcore.episode import Episode, EpisodeDefault
from mojograsp.simcore.record_data import RecordData, RecordDataDefault
from mojograsp.simcore.replay_buffer import ReplayBuffer, ReplayBufferDefault
from mojograsp.simcore.phase_manager import PhaseManager
from mojograsp.simcore.state import State, StateDefault
from mojograsp.simcore.reward import Reward, RewardDefault
from mojograsp.simcore.action import Action, ActionDefault
# python imports
import time
import os
from abc import ABC, abstractmethod
import logging
from PIL import Image
from torch.utils.tensorboard import SummaryWriter


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
                 episode: Episode = EpisodeDefault(), record_data: RecordData = RecordDataDefault()):
        """
            Constructor passes in the environment, episode and record data objects and simmanager parameters.

            :param num_episodes: Number of episodes to run.
            :param env: :func:`~mojograsp.simcore.environment.Environment` object
            :param episode: :func:`~mojograsp.simcore.episode.Episode` object
            :param record_data: :func:`~mojograsp.simcore.record_data.RecordData` object
            :param state: :func:`~mojograsp.simcore.state.State` object
            :param action: :func:`~mojograsp.simcore.action.Action` object
            :param reward: :func:`~mojograsp.simcore.reward.Reward` object
            :type num_episodes: int
            :type env: :func:`~mojograsp.simcore.environment.Environment`
            :type episode: :func:`~mojograsp.simcore.episode.Episode`
            :type record: :func:`~mojograsp.simcore.record_data.RecordData`
            :type state: :func:`~mojograsp.simcore.state.State`
            :type action: :func:`~mojograsp.simcore.action.Action`
            :type reward: :func:`~mojograsp.simcore.reward.Reward`
            """
        self.num_episodes = num_episodes
        self.env = env
        self.episode = episode
        self.record = record_data
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
                    count = 0
                    while not done:
                        self.phase_manager.current_phase.pre_step()
                        self.phase_manager.current_phase.execute_action()
                        self.env.step()
                        self.phase_manager.current_phase.post_step()
                        self.record.record_timestep()
#                        img = p.getCameraImage(640, 480, renderer=p.ER_BULLET_HARDWARE_OPENGL)
#                        img = Image.fromarray(img[2])
#                        img.save('/home/mechagodzilla/mojo-grasp/demos/expert_demo/data/frame_'+str(count)+'.png')
                        count += 1
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


class SimManagerRL(SimManager):
    """
    SimManagerRL class, runs through all episodes and uses the passed in
    mojograsp classes to call the necessary functions. This default will work on most RL applications.  
    """

    def __init__(self, num_episodes: int = 1, env: Environment = EnvironmentDefault(),
                 episode: Episode = EpisodeDefault(), record_data: RecordData = RecordDataDefault(),
                 replay_buffer: ReplayBuffer = ReplayBufferDefault, state: State = StateDefault,
                 action: Action = ActionDefault, reward: Reward = RewardDefault):
        """
        Constructor passes in the environment, episode and record data objects and simmanager parameters.

        :param num_episodes: Number of episodes to run.
        :param env: :func:`~mojograsp.simcore.environment.Environment` object
        :param episode: :func:`~mojograsp.simcore.episode.Episode` object
        :param record_data: :func:`~mojograsp.simcore.record_data.RecordData` object
        :param state: :func:`~mojograsp.simcore.state.State` object
        :param action: :func:`~mojograsp.simcore.action.Action` object
        :param reward: :func:`~mojograsp.simcore.reward.Reward` object
        :param replay_buffer: :func:`~mojograsp.simcore.replay_buffer.ReplayBuffer` object
        :type num_episodes: int
        :type env: :func:`~mojograsp.simcore.environment.Environment`
        :type episode: :func:`~mojograsp.simcore.episode.Episode`
        :type record: :func:`~mojograsp.simcore.record_data.RecordData`
        :type state: :func:`~mojograsp.simcore.state.State`
        :type action: :func:`~mojograsp.simcore.action.Action`
        :type reward: :func:`~mojograsp.simcore.reward.Reward`
        :type replay_buffer: :func:`~mojograsp.simcore.replay_buffer.ReplayBuffer`
        """
        self.state = state
        self.action = action
        self.reward = reward
        self.num_episodes = num_episodes
        self.env = env
        self.episode = episode
        self.record = record_data
        self.replay_buffer = replay_buffer
        self.phase_manager = PhaseManager()
        self.episode_number = 0
        self.writer = SummaryWriter()

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
            S = None
            A = None
            R = None
            S2 = None
            E = self.episode_number

            for i in range(self.num_episodes):
                self.episode_number += 1
                self.episode.setup()
                self.env.setup()
                self.phase_manager.set_exit_flag(False)
                self.phase_manager.setup()

                timestep_number = 0
                while self.phase_manager.get_exit_flag() == False:
                    self.phase_manager.current_phase.setup()
                    done = False
                    logging.info("CURRENT PHASE: {}".format(
                        self.phase_manager.current_phase.name))
                    while not done:
                        timestep_number += 1
                        self.phase_manager.current_phase.pre_step()
                        self.state.set_state()
                        S = self.state.get_state()
                        self.phase_manager.current_phase.execute_action()
                        A = self.action.get_action()
                        self.env.step()
                        done = self.phase_manager.current_phase.exit_condition()
                        self.phase_manager.current_phase.post_step()
                        self.record.record_timestep()
                        R = self.reward.get_reward()
                        self.state.set_state()
                        S2 = self.state.get_state()
                        E = self.episode_number
                        transition = (S, A, R, S2, E)
                        self.replay_buffer.add_timestep(transition)
                        done = self.phase_manager.current_phase.exit_condition()
                        self.phase_manager.current_phase.controller.train_policy()
#                        img = p.getCameraImage(640, 480, renderer=p.ER_BULLET_HARDWARE_OPENGL)
#                        img = Image.fromarray(img[2])
#                        img.save('/home/orochi/mojo/mojo-grasp/demos/rl_demo/data/vizualization/episode_' + num_string(self.episode_number) + '_frame_'+ num_string(timestep_number)+'.png')
                    self.phase_manager.get_next_phase()
                self.record.record_episode()
                self.record.save_episode()
                self.episode.post_episode()
                self.env.reset()
                self.writer.add_scalar('rewards/average reward', self.replay_buffer.get_average_reward(
                    400), self.episode_number/self.num_episodes)
                self.writer.add_scalar('rewards/min reward', self.replay_buffer.get_min_reward(400),
                                       self.episode_number / self.num_episodes)
                self.writer.add_scalar('rewards/max reward', self.replay_buffer.get_max_reward(400),
                                       self.episode_number / self.num_episodes)
            self.record.save_all()
            # self.replay_buffer.save_buffer('./data/temp_buffer.pkl')
        else:
            logging.warn("No Phases have been added")
        # print("COMPLETED")

    def stall(self):
        super().stall()
