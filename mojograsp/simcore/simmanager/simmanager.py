import time
import os
import shutil
import pybullet as p
import pybullet_data
from . import episode
from mojograsp.simcore.simmanager.State.State_Metric.state_metric_base import StateMetricBase
from . import phase
from . import phasemanager
from . import controller_base
from mojograsp.simcore.simmanager.Reward import reward_base
from mojograsp.simcore.simmanager.State.state_space_base import StateSpaceBase
from mojograsp.simcore.simmanager.Action.action_class import Action
from mojograsp.simcore.simmanager.record_episode import RecordEpisode
from mojograsp.simcore.simmanager.record_timestep import RecordTimestep
from mojograsp.simcore.simmanager.replay_buffer import ReplayBuffer


class SimManagerBase:
    # TODO: fill in with relevant classes
    def __init__(self, num_episodes=1, sim_timestep=(1. / 240.), episode_timestep_length=1,
                 episode_configuration=None, rl=False, data_directory_path=None, replay_episode_file=None):
        # initializes phase dictionary and other variables we will need
        self.current_phase = None
        self.starting_phase = None
        # Redundant maybe?
        self.rl = rl
        self.env = None
        # sets episode configuration object and checks if it is none, if it is we create our own empty one
        # need this for stepping later
        self.episode_configuration = episode_configuration  # TODO: add episode config functions straight to simmanager
        if (self.episode_configuration == None):
            self.episode_configuration = episode.Episode()

        # variables to keep track of episodes and timestep lengths
        self.num_episodes = num_episodes
        self.episode_timestep_length = episode_timestep_length  # TODO: TimeParam class goes here I think
        self.sim_timestep = sim_timestep

        #replay buffer
        # self.replay_expert = ReplayBuffer(episodes_file=replay_episode_file)
        # self.replay_agent = ReplayBuffer()

        self.data_path = data_directory_path
        self.create_data_directorys()


        self.phase_manager = phasemanager.PhaseManager()

        self.state_space = None
        self.reward_space = None
        # physics server setup, in the future needs arguments
        self.setup()

    def setup(self):
        """
        Physics server setup.

        Simulator specific, should be defined by user.
        """
        pass

    def create_data_directorys(self):
        episode = self.data_path+'/episodes'
        timesteps = self.data_path+'/timesteps'
        if not os.path.exists(episode):
            os.makedirs(episode)
        else:
            shutil.rmtree(episode)
            os.makedirs(episode)
        if not os.path.exists(timesteps):
            os.makedirs(timesteps)
        else:
            shutil.rmtree(timesteps)
            os.makedirs(timesteps)

    def stall(self):
        """
        Prevent simulator from closing.

        Simulator specific, should be defined by user.
        """
        pass

    # adds a phase to our phase dictionary
    def add_phase(self, phase_name, phase_object, start=False):
        self.phase_manager.add_phase(phase_name, phase_object, start)

    def step(self):
        pass

    def run(self):
        pass


# TODO: make this SimManager_bullet class
class SimManagerPybullet(SimManagerBase):

    def __init__(self, num_episodes=1, sim_timestep=(1. / 240.), episode_timestep_length=1, episode_configuration=None,
                 rl=False, data_directory_path=None, replay_episode_file=None):
        super(SimManagerPybullet, self).__init__(num_episodes=num_episodes, sim_timestep=sim_timestep,
                                                 episode_timestep_length=episode_timestep_length,
                                                 episode_configuration=episode_configuration, rl=rl, data_directory_path=data_directory_path,
                                                 replay_episode_file=replay_episode_file)

    # physics server setup, TODO: in the future needs arguments
    def setup(self):
        """
        Physics server setup.
        """
        # self.physics_client = p.connect(p.DIRECT)
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        self.plane_id = p.loadURDF("plane.urdf")

    # Prevents pybullet from closing
    def stall(self):
        """
        Prevents pybullet from closing
        """
        while p.isConnected():
            time.sleep(1)

    def add_env(self, env):
        """
        Adds the working environment for the simulator which handles stepping through, taking actions, etc
        :param env: Either a gym environment or self-specified environment
        """
        self.env = env
        phase.Phase._sim = self.env
        StateMetricBase._sim = self.env
        controller_base.ControllerBase._sim = self.env
        reward_base.RewardBase._sim = self.env
        StateSpaceBase._sim = self.env
        Action._sim = self.env
        RecordEpisode._sim = self.env
        RecordTimestep._sim = self.env

    def run(self):
        print("RUNNING PHASES: {}".format(self.phase_manager.phase_dict))
        training_phase = self.phase_manager.phase_dict['move rl']
        # # Evaluation of trained policy: (Uncomment below line)

        training_phase.controller.load('saved_weights')
        #resets episode settings, runs episode setup and sets the current phase
        for i in range(self.num_episodes):
            self.env.reset()
            self.episode_configuration.reset()
            self.episode_configuration.setup()
            self.phase_manager.exit_flag = False
            self.phase_manager.start_phases()
            # record_episode = RecordEpisode(identifier='cube_rl', data_path=self.data_path)

            #for every phase in the dictionary we step until the exit condition is met
            while self.phase_manager.exit_flag == False:
                self.phase_manager.setup_phase()
                phase_step_count = 0
                done = False

                #while exit condition is not met call step
                # print("CURRENT PHASE: {}".format(self.phase_manager.current_phase.name))
                while not done:

                    self.phase_manager.current_phase.curr_action = self.phase_manager.current_phase.controller.select_action()
                    self.episode_configuration.episode_pre_step()
                    observation, reward, _, info = self.env.step(self.phase_manager.current_phase)
                    # print("Timestep Reward: {}".format(reward))
                    self.episode_configuration.episode_post_step()
                    done = self.phase_manager.current_phase.phase_exit_condition(phase_step_count)
                    phase_step_count += 1
                    self.env.curr_timestep += 1
                    # record_timestep = RecordTimestep(self.phase_manager.current_phase, data_path=self.data_path)
                    # record_episode.add_timestep(record_timestep)
                    # # record_timestep.save_timestep_as_csv()

                #after exit condition is met we get the next phase name and set current phase to the specified value
                self.phase_manager.get_next_phase()

                if self.phase_manager.exit_flag is True:
                    # Training of network (Everything inside if statement. Comment while evaluating)
                    # if i > 20:
                        # print("Starting Training ", i)
                        # training_phase.controller.train(training_phase.terminal_step, expert_replay_buffer=self.replay_expert,
                        #                                 replay_buffer=self.replay_agent)
                        # training_phase.controller.train_batch(max_episode_num=training_phase.terminal_step, episode_num=i, update_count=1, expert_replay_buffer=self.replay_expert,
                        #                                       replay_buffer=self.replay_agent)
                    break

            # print("Episode Reward: {}".format(reward))
            if not (i % 100):
                print(i)
            # self.replay_expert.add_episode(record_episode)
            # self.replay_agent.add_episode(record_episode, i)
            # record_episode.save_episode_as_csv(episode_number=i)
            #TODO needs to be in episode class instead of here
        print("Saving replay buffer...")
        # self.replay_expert.save_replay_buffer('expert_replay_{}'.format(i))
        # self.replay_agent.save_replay_buffer('expert_replay_{}'.format(i))
        print("Saving weights...")
        # training_phase.controller.save('saved_weights')
        print("Done!")
