import pybullet as p

from mojograsp.simcore.phase import Phase

import time
import logging


class PhaseManager:
    def __init__(self):
        self.phase_dict = {}
        self.starting_phase = None
        self.current_phase = None
        self.exit_flag = None

    def add_phase(self, phase_name: str, phase: Phase, start=False):
        self.phase_dict[phase_name] = phase

        # if start flag set or only phase given we set it as starting phase
        if len(self.phase_dict) == 1 or start is True:
            self.starting_phase = phase
            logging.info("Added phase {} as starting phase".format(phase_name))
        logging.info("Added phase: {}".format(phase_name))

    def setup(self):
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

    def get_exit_flag(self) -> bool:
        return self.exit_flag

    def set_exit_flag(self, val: bool = False):
        self.exit_flag = val
