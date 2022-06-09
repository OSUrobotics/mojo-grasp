import pybullet as p

from mojograsp.simcore.phase import Phase

import time
import logging


class PhaseManager:
    """Phase Manager Class"""

    def __init__(self):
        """
        Constructor initializes the phase dictionary, starting phase, current phase and 
        exit flag to Empty and None.
        """
        self.phase_dict = {}
        self.starting_phase = None
        self.current_phase = None
        self.exit_flag = None

    def add_phase(self, phase_name: str, phase: Phase, start=False):
        """
        Method takes in a phase name, phase object and a boolean to specify whether it is the starting phase.
        Adds the phase to the phase dictionary in the style PHASE_NAME: PHASE_OBJECT.

        :param phase_name: Name of the phase being added.
        :param phase: :func:`~mojograsp.simcore.phase.Phase` object.
        :param start: If True then the phase given will be the starting phase.
        :type phase_name: str
        :type phase: :func:`~mojograsp.simcore.phase.Phase` 
        :type start: bool
        """
        self.phase_dict[phase_name] = phase
        # if start flag set or only phase given we set it as starting phase
        if len(self.phase_dict) == 1 or start is True:
            self.starting_phase = phase
            logging.info("Added phase {} as starting phase".format(phase_name))
        logging.info("Added phase: {}".format(phase_name))

    def setup(self):
        """Method called before starting episodes by :func:`~mojograsp.simcore.sim_manager.SimManager`, sets the current phase to the starting phase"""
        self.current_phase = self.starting_phase

    def get_next_phase(self):
        """
        Method called by :func:`~mojograsp.simcore.sim_manager.SimManager` to get the next phase
        after the current one has completed. If no phase is given or phase name does not exist
        in the phase dictionary then the phase manager terminates.
        """
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
        """Gets the internal exit flag so that :func:`~mojograsp.simcore.sim_manager.SimManager` can check if all phases are complete"""
        return self.exit_flag

    def set_exit_flag(self, val: bool = False):
        """Allows :func:`~mojograsp.simcore.sim_manager.SimManager` to manually set the exit flag after a run is completed for the next episode."""
        self.exit_flag = val
