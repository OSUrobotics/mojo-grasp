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

    def get_next_phase(self):
        next_phase = self.current_phase.next_phase()
        if next_phase != None:
            try:
                self.current_phase = self.phase_dict[next_phase]
            except:
                self.exit_flag = True
                print("Error: Could not find next phase " + str(next_phase))
        # if phase is none we break the for loop early
        else:
            self.exit_flag = True
