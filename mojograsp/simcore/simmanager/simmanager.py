import time
import pybullet as p
import pybullet_data
# import episode
# import environment
from . import episode
from . import environment
import gym
import gym_env_files
from mojograsp.simcore.simmanager.State.State_Metric.state_metric_base import StateMetricBase
from . import phase
from . import phasemanager

class SimManager_base:
    # TODO: fill in with relevant classes
    def __init__(self, num_episodes=1, sim_timestep=(1. / 240.), episode_timestep_length=1,
                 episode_configuration=None, rl=False):
        # initializes phase dictionary and other variables we will need
        self.phase_dict = {}
        self.current_phase = None
        self.starting_phase = None
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

        self.phase_manager = phasemanager.PhaseManager(episode_timestep_length=episode_timestep_length,
                                                        sim_timestep=sim_timestep)

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
        self.phase_dict[phase_name] = phase_object

        # if start flag set or only phase given we set it as starting phase
        if (len(self.phase_dict) == 1 or start == True):
            self.starting_phase = phase_object
        # TODO: should we pass phase list to phasemanager each time we run phases,
        #  or should we bake it onto phasemanager?
        self.phase_manager.add_phase_dict(self.phase_dict, self.starting_phase)

        print("Added Phase")

    def step(self):
        pass

    def run(self):
        pass


# TODO: make this SimManager_bullet class
class SimManager_Pybullet(SimManager_base):

    def __init__(self, num_episodes=1, sim_timestep=(1. / 240.), episode_timestep_length=1, episode_configuration=None,
                 rl=False):
        super(SimManager_Pybullet, self).__init__(num_episodes=num_episodes, sim_timestep=sim_timestep,
                                                  episode_timestep_length=episode_timestep_length,
                                                  episode_configuration=episode_configuration, gym=gym)

    #physics server setup, TODO: in the future needs arguments
    def setup(self):
        """
        Physics server setup.
        """
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0,0,-10)
        plane_id = p.loadURDF("plane.urdf")

    # Prevents pybullet from closing
    def stall(self):
        """
        Prevents pybullet from closing
        """
        while p.isConnected():
            time.sleep(1)

    def add_env(self, env):
        self.env = env
        phase.Phase._sim = self.env
        StateMetricBase._sim = self.env

    def add_state_space(self, state_space):
        self.state_space = state_space

    def add_rewards(self, reward):
        self.reward_space = reward

    #rough outline of simstep where phase functions called and pybullet stepped n times
    def step(self):  # TODO: rename to update?
        #take episode pre step action
        self.episode_configuration.episode_pre_step()

        self.phase_manager.step(self.episode_timestep_length, self.sim_timestep)

        #after 1 episode step we call the episode post step function
        self.episode_configuration.episode_post_step()


    #Rough outline of run, runs for set amount of episodes where within each episode all phases are run until their exit condition is met
    def run(self):
        print("RUNNING")

        #resets episode settings, runs episode setup and sets the current phase
        for i in range(self.num_episodes):
            self.env.reset()
            self.episode_configuration.reset()
            self.episode_configuration.setup()

            self.phase_manager.run_phases()





