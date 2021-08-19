import time
import pybullet as p


class PhaseManager:
    def __init__(self):
        self.phase_dict = {}
        self.starting_phase = None
        self.current_phase = None
        self.last_phase = None
        self.exit_flag = None

    def add_phase(self, phase_name, phase_object, start=False):
        self.phase_dict[phase_name] = phase_object

        # if start flag set or only phase given we set it as starting phase
        if len(self.phase_dict) == 1 or start is True:
            self.starting_phase = phase_object
        print("Added Phase")

    def start_phases(self):
        self.current_phase = self.starting_phase
        return self.current_phase

    def get_next_phase(self):
        next_phase = self.current_phase.phase_complete()
        if (next_phase != None):
            try:
                self.current_phase = self.phase_dict[next_phase]
                # return self.current_phase
            except:
                self.exit_flag = True
                print("Error: Could not find next phase " + str(next_phase))
        # if phase is none we break the for loop early
        else:
            self.exit_flag = True
            # return next_phase


    def setup_phase(self):
        self.current_phase.setup()

# class PhaseManager:
#     # TODO: get functional shell down
#     def __init__(self, episode_timestep_length, sim_timestep):
#         self.phases = None  # dictionary with phase objects in it
#         self.starting_phase = None
#         self.current_phase = None
#
#         self.episode_timestep_length = episode_timestep_length
#         self.sim_timestep = sim_timestep
#
#     def add_phase_dict(self, phase_dict, starting_phase):
#         self.phases = phase_dict
#         self.starting_phase = starting_phase
#
    # def run_phases(self):
    #     self.current_phase = self.starting_phase
    #     exit_flag = False
    #
    #     # for every phase in the dictionary we step until the exit condition is met
    #     while (exit_flag == False):  # TODO: running phases now goes into phasemanager
    #         self.current_phase.setup()
    #         step_count = 0
    #         done = False
    #         #@anjali
    #         while not done:
    #             pass
    #         #@anjali
    #
    #         # while exit condition is not met call step
    #         while self.current_phase.phase_exit_condition(step_count) == False:
    #             self.step()
    #             step_count += 1
    #
    #         # after exit condition is met we get the next phase name and set current phase to the specified value
    #         next_phase = self.current_phase.phase_complete()
    #         if (next_phase != None):
    #             try:
    #                 self.current_phase = self.phase_dict[next_phase]
    #             except:
    #                 exit_flag = True
    #                 print("Error: Could not find next phase " + str(next_phase))
    #         # if phase is none we break the for loop early
    #         else:
    #             exit_flag = True
    #             break

    # def step(self):
    #     # select action before episode step
    #     action = self.current_phase.select_action()
    #
    #     # phase prestep and execute called
    #     self.current_phase.pre_step()  # TODO: do we need this? What about user functions?
    #     self.current_phase.execute_action()
    #
    #     # simulator timesteps equaling one episode timestep
    #     for i in range(self.episode_timestep_length):  # TODO: pass this in through constructor
    #         # Pybullet stepped
    #         p.stepSimulation()
    #         time.sleep(self.sim_timestep)
    #
    #     # phase post step called
    #     self.current_phase.post_step()

