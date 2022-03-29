# pybullet imports
import pybullet as p
import pybullet_data
# mojograsp module imports
from mojograsp.simcore.environment import Environment
from mojograsp.simcore.phase import Phase
from mojograsp.simcore.state import State
from mojograsp.simcore.episode import Episode
from mojograsp.simcore.record_data import RecordData
from mojograsp.simcore.replay_buffer import ReplayBuffer
# python imports
import time
import os
# alsdkf
from abc import ABC, abstractmethod


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
        '''Prevents pybullet from closing'''
        while p.isConnected():
            time.sleep(1)


class SimManagerDefault(SimManager):

    def __init__(self, num_episodes: int = 1, sim_timestep: float = (1. / 240.), data_path: str = None, gui: Boolean = True,
                 environment: Environment = None, state: State = None, episode: Episode = None, record: RecordData = None,
                 replay_buffer: ReplayBuffer = None):
        pass

    def setup(self):
        '''Physics server setup.'''
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        self.plane_id = p.loadURDF("plane.urdf")

    def inject_env(self):
        pass

    def run(self):
        print("RUNNING PHASES: {}".format(self.phase_manager.phase_dict))
        # User Function 1
        self.user_func.pre_run(self.phase_manager.phase_dict)

        # resets episode settings, runs episode setup and sets the current phase
        for i in range(self.num_episodes):
            self.env.reset()
            self.episode_configuration.reset()
            self.episode_configuration.setup()
            self.phase_manager.exit_flag = False
            self.phase_manager.start_phases()
            record_episode = RecordEpisode(
                identifier='cube', data_path=self.data_path)
            self.user_func.pre_phaseloop()

            # for every phase in the dictionary we step until the exit condition is met
            while self.phase_manager.exit_flag == False:
                self.phase_manager.setup_phase()
                phase_step_count = 0
                done = False
                self.user_func.pre_phase()

                # while exit condition is not met call step
                # print("CURRENT PHASE: {}".format(self.phase_manager.current_phase.name))
                while not done:
                    self.phase_manager.current_phase.curr_action = self.phase_manager.current_phase.controller.select_action()
                    self.episode_configuration.episode_pre_step()
                    observation, reward, _, info = self.env.step(
                        self.phase_manager.current_phase)
                    self.episode_configuration.episode_post_step()
                    self.phase_manager.current_phase.post_step()
                    done = self.phase_manager.current_phase.phase_exit_condition(
                        phase_step_count)
                    phase_step_count += 1
                    self.env.curr_timestep += 1
                    record_timestep = RecordTimestep(
                        self.phase_manager.current_phase, data_path=self.data_path)
                    record_episode.add_timestep(record_timestep)

                self.user_func.post_phase()
                # after exit condition is met we get the next phase name and set current phase to the specified value
                self.phase_manager.get_next_phase()

                if self.phase_manager.exit_flag is True:
                    # User function 2
                    self.user_func.post_lastphase(
                        data=[i, self.replay_expert, self.replay_agent])
                    break

            # print("Episode Reward: {}".format(reward))

            # User function 3
            self.user_func.post_phaseloop(data=[i, reward])

            if self.replay_agent is not None:
                self.replay_agent.add_episode(record_episode, i)
            if self.replay_expert is not None:
                self.replay_expert.add_episode(record_episode, i)

            record_episode.save_episode_as_csv(episode_number=i)

            # TODO needs to be in episode class instead of here

        print("Saving replay buffer...")
        if self.replay_agent is not None:
            self.replay_agent.save_replay_buffer('agent_replay_{}'.format(i))
        if self.replay_expert is not None:
            self.replay_expert.save_replay_buffer('expert_replay_{}'.format(i))

        # User function 4
        self.user_func.post_run()
        print("Done!")
