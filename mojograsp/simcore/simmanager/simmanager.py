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
        #@anjali
        # self.phase_dict[phase_name] = phase_object
        #
        # # if start flag set or only phase given we set it as starting phase
        # if (len(self.phase_dict) == 1 or start == True):
        #     self.starting_phase = phase_object
        # # TODO: should we pass phase list to phasemanager each time we run phases,
        # #  or should we bake it onto phasemanager?
        # self.phase_manager.add_phase_dict(self.phase_dict, self.starting_phase)
        #
        # print("Added Phase")

        self.phase_manager.add_phase(phase_name, phase_object, start)
        #@anjali

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
                                                  episode_configuration=episode_configuration, rl=rl)

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

    def run(self):
        print("RUNNING PHASES: {}".format(self.phase_manager.phase_dict))

        #resets episode settings, runs episode setup and sets the current phase
        for i in range(self.num_episodes):
            self.env.reset()
            self.episode_configuration.reset()
            self.episode_configuration.setup()
            self.phase_manager.exit_flag = False
            self.phase_manager.start_phases()
            done = False

            #for every phase in the dictionary we step until the exit condition is met
            while self.phase_manager.exit_flag == False:
                self.phase_manager.setup_phase()
                step_count = 0
                done = False

                #while exit condition is not met call step
                print("CURRENT PHASE: {}".format(self.phase_manager.current_phase.name))
                while not done:
                    print(step_count, i)
                    self.phase_manager.current_phase.curr_action = self.phase_manager.current_phase.controller.select_action()
                    self.episode_configuration.episode_pre_step()
                    observation, reward, _, info = self.env.step(self.phase_manager.current_phase)
                    self.episode_configuration.episode_post_step()
                    done = self.phase_manager.current_phase.phase_exit_condition(step_count)
                    step_count += 1

                #after exit condition is met we get the next phase name and set current phase to the specified value
                self.phase_manager.get_next_phase()
                if self.phase_manager.exit_flag is True:
                    break
