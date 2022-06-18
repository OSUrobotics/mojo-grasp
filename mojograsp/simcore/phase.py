from abc import ABC, abstractmethod


class Phase(ABC):
    """Phase Abstract Base Class"""
    @abstractmethod
    def __init__(self):
        """
        Constructor should be used to pass any other objects or 
        data that the Phase class will need to use.
        Usually this will be a manipulator or other mojograsp objects.
        """
        pass

    @abstractmethod
    def setup(self):
        """
        This method is called by :func:`~mojograsp.simcore.sim_manager.SimManager` once before the phase begins.
        """
        pass

    @abstractmethod
    def execute_action(self):
        """
        This method is called every step by :func:`~mojograsp.simcore.sim_manager.SimManager` while the currrent phase is active. Operations
        such as moving joints and executing actions should be done here. 
        """
        pass

    @abstractmethod
    def exit_condition(self) -> bool:
        """
        This method is called by :func:`~mojograsp.simcore.sim_manager.SimManager` every Phase iteration
        and returns either a Boolean, signifying if the current phase should end or
        continue looping. Returning True signifies that the current phase should end and function
        :func:`~mojograsp.simcore.phase.Phase.next_phase` should be called. Any exit conditions for
        a phase should be kept here.

        :return: True (exit phase) or False (continue phase).
        :rtype: bool
        """
        pass

    @abstractmethod
    def next_phase(self) -> str:
        """
        This method is called by :func:`~mojograsp.simcore.sim_manager.SimManager` once a phase has returned 
        True for its exit_condition(). Returns a string specifying the next phase that the
        :func:`~mojograsp.simcore.phase_manager.PhaseManager` should transition to next. To signify the end of 
        an episode return None. 

        :return: The name of the next phase or None
        :rtype: str or None
        """
        pass

    @abstractmethod
    def pre_step(self):
        """
        This method is called before every step by :func:`~mojograsp.simcore.sim_manager.SimManager`
        while the currrent phase is active. 
        """
        pass

    @abstractmethod
    def post_step(self):
        """
        This method is called after every step by :func:`~mojograsp.simcore.sim_manager.SimManager`
        while the currrent phase is active.
        """
        pass
