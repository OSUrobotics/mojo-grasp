from abc import ABC, abstractmethod

from mojograsp.simobjects.object_base import ObjectBase


class State(ABC):
    """State Abstract Base Class"""
    @abstractmethod
    def __init__(self):
        """
        Constructor should be used to pass any other objects or 
        data that a State will need to function. Usually this will be a manipulator 
        or other mojograsp objects.
        """
        self.current_state = {}
        pass

    @abstractmethod
    def set_state(self):
        """
        Method should be used to set the class variable self.current_state to represent the current state. 
        It should be called by the pre_step() method of a user defined phase. 
        This can be set however you would like, as long as self.current_action is updated so that get_action is returns
        properly. This could include joint angles, locations, etc.
        """
        self.current_state = {}

    @abstractmethod
    def get_state(self) -> dict:
        """
        Method should return a dictionary that represents the current state of the simulator environment
        BEFORE it is executed and the sim is stepped. The return value is class variable 
        self.current_state which is updated by set_state(). 

        :return: Dictionary containing the representation of the current state as a dictionary
        :rtype: Dictionary with format {string: **ANY TYPE**}.
        """
        return self.current_state


class StateDefault(State):
    """
    Default State Class that is used when the user does not need or wish to use the Action class
    but it may still be called by other classes such as :func:`~mojograsp.simcore.record_data.RecordData` 
    and :func:`~mojograsp.simcore.replay_buffer.ReplayBufferDefault`.
    """

    def __init__(self, objects: list = None):
        """
        Default placeholder constructor optionally takes in a list of objects, if no list is provided it defaults
        to None. 

        :param objects: list of mojograsp objects.
        :type objects: list
        """
        super().__init__()
        self.objects = objects

    def set_state(self):
        """
        Default method that sets self.current_state to either get_data() for the object or an empty dictionary
        """
        if self.objects:
            data_dict = {}
            for i in self.objects:
                data_dict[i.name] = i.get_data()
            self.current_state = data_dict
        else:
            self.current_state = {}

    def get_state(self) -> dict:
        """
        Default method will return a dictionary containing the the get_data() return value for every object
        in the objects list. If no objects are given then it returns an empty dictionary.

        :return: Dictionary containing the representation of the current simulator state or an empty dictionary.
        :rtype: dict
        """
        return self.current_state
