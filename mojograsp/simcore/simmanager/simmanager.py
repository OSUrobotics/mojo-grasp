import time
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
from mojograsp.simcore.simmanager.record_episode_base import RecordEpisodeBase
from mojograsp.simcore.simmanager.record_timestep_base import RecordTimestepBase
from mojograsp.simcore.simmanager.record_episode import RecordEpisode
from mojograsp.simcore.simmanager.record_timestep import RecordTimestep


class SimManagerBase:
    # TODO: fill in with relevant classes
    def __init__(self, num_episodes=1, sim_timestep=(1. / 240.), episode_timestep_length=1,
                 episode_configuration=None, rl=False):
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
                 rl=False):
        super(SimManagerPybullet, self).__init__(num_episodes=num_episodes, sim_timestep=sim_timestep,
                                                 episode_timestep_length=episode_timestep_length,
                                                 episode_configuration=episode_configuration, rl=rl)

    # physics server setup, TODO: in the future needs arguments
    def setup(self):
        """
        Physics server setup.
        """
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
        RecordEpisodeBase._sim = self.env
        RecordTimestepBase._sim = self.env

    def run(self):
        print("RUNNING PHASES: {}".format(self.phase_manager.phase_dict))

        #resets episode settings, runs episode setup and sets the current phase
        for i in range(self.num_episodes):
            self.env.reset()
            self.episode_configuration.reset()
            self.episode_configuration.setup()
            self.phase_manager.exit_flag = False
            self.phase_manager.start_phases()
            record_episode = RecordEpisode(identifier='cube_{}'.format(i))

            #for every phase in the dictionary we step until the exit condition is met
            while self.phase_manager.exit_flag == False:
                self.phase_manager.setup_phase()
                phase_step_count = 0
                done = False

                #while exit condition is not met call step
                print("CURRENT PHASE: {}".format(self.phase_manager.current_phase.name))
                while not done:
                    print(self.env.curr_timestep, i)
                    self.phase_manager.current_phase.curr_action = self.phase_manager.current_phase.controller.select_action()
                    self.episode_configuration.episode_pre_step()
                    observation, reward, _, info = self.env.step(self.phase_manager.current_phase)
                    self.episode_configuration.episode_post_step()
                    done = self.phase_manager.current_phase.phase_exit_condition(phase_step_count)
                    phase_step_count += 1
                    self.env.curr_timestep += 1
                    record_timestep = RecordTimestep(self.phase_manager.current_phase)
                    record_episode.add_timestep(record_timestep)
                    # record_timestep.save_timestep_as_csv()

                #after exit condition is met we get the next phase name and set current phase to the specified value
                self.phase_manager.get_next_phase()
                training_phase = self.phase_manager.phase_dict['move rl']
                training_phase.controller.train(training_phase.terminal_step, None)
                if self.phase_manager.exit_flag is True:
                    break
            # record_episode.save_episode_as_csv()
