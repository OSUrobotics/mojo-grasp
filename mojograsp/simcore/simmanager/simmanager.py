import time
import pybullet as p
import pybullet_data
# import episode
# import environment
from . import episode
from . import environment
import gym
import gym_env_files


class SimManager:

    def __init__(self, num_episodes=1, sim_timestep=(1. / 240.), episode_timestep_length=1, episode_configuration=None,
                 rl=False):
        #initializes phase dictionary and other variables we will need
        self.phase_dict = {}
        print("###########Not stuck here1#########")
        self.current_phase = None
        print("Not stuck here2")
        self.starting_phase = None
        print("Not stuck here3")
        self.rl = rl
        print("Not stuck here4")
        if self.rl:
            self.env = gym.make("ihm-v1")
            self.env.reset()
        else:
            self.env = environment.Environment()
        print("Not stuck here5")

        #sets episode configuration object and checks if it is none, if it is we create our own empty one
        #need this for stepping later
        self.episode_configuration = episode_configuration
        if(self.episode_configuration == None):
            print("Not stuck here6")
            self.episode_configuration = episode.Episode()
            print("Not stuck here7")

        #variables to keep track of episodes and timestep lengths
        self.num_episodes = num_episodes
        self.episode_timestep_length = episode_timestep_length
        self.sim_timestep = sim_timestep

        #physics server setup, in the future needs arguments
        self.setup()


    #physics server setup, in the future needs arguments
    def setup(self):
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0,0,-10)
        plane_id = p.loadURDF("plane.urdf")

    #Prevents pybullet from closing
    def stall(self):
        while p.isConnected():
            time.sleep(1)

    def add_state_space(self, state_space):
        self.state_space = state_space

    def add_rewards(self, reward):
        self.reward_space = reward

    #adds a phase to our phase dictionary
    def add_phase(self, phase_name, phase_object, start=False):
        self.phase_dict[phase_name] = phase_object

        #if start flag set or only phase given we set it as starting phase
        if(len(self.phase_dict) == 1 or start == True):
            self.starting_phase = phase_object

        print("Added Phase: {}".format(phase_name))

    #Rough outline of run, runs for set amount of episodes where within each episode all phases are run until their exit condition is met
    def run(self):
        print("RUNNING PHASES: {}".format(self.phase_dict))

        #resets episode settings, runs episode setup and sets the current phase
        for i in range(self.num_episodes):
            self.env.reset()
            self.episode_configuration.reset()
            self.episode_configuration.setup()
            self.current_phase = self.starting_phase
            exit_flag = False
            done = False

            #for every phase in the dictionary we step until the exit condition is met
            while(exit_flag == False):
                self.current_phase.setup()
                step_count = 0
                done = False

                #while exit condition is not met call step
                print("CURRENT PHASE: {}".format(self.current_phase.name))
                while not done:
                    print(step_count, self.state_space)
                    self.current_phase.action = self.current_phase.controller.select_action()
                    val = [self.current_phase, self.state_space, self.reward_space]
                    observation, reward, done, info = self.env.step(val)
                    done = self.current_phase.phase_exit_condition(step_count)
                    step_count+=1

                #after exit condition is met we get the next phase name and set current phase to the specified value
                next_phase = self.current_phase.phase_complete()
                if(next_phase != None):
                    try:
                        self.current_phase = self.phase_dict[next_phase]
                    except:
                        exit_flag = True
                        print("Error: Could not find next phase " + str(next_phase))
                #if phase is none we break the for loop early
                else:
                    exit_flag = True
                    break

if __name__ == '__main__':
    # setting up simmanager/physics server
    # env = gym.make('ihm-v1')
    # print("a")
    manager = SimManager(rl=True)


