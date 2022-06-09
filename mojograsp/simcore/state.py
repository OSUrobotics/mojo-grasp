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
        pass

    @abstractmethod
    def get_state(self) -> dict:
        """
        Method should return a dictionary that represents the current state of the simulator
        BEFORE an action is taken and the sim is stepped. This could include joint angles,
        object locations, etc.

        :return: Dictionary containing the representation of the current simulator state
        :rtype: Dictionary with format {string: **ANY TYPE**}.
        """
        data_dict = {}
        return data_dict


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

    def get_state(self) -> dict:
        """
        Default method will return a dictionary containing the the get_data() return value for every object
        in the objects list. If no objects are given then it returns an empty dictionary.

        :return: Dictionary containing the representation of the current simulator state or an empty dictionary.
        :rtype: dict
        """
        if self.objects:
            data_dict = {}
            for i in self.objects:
                data_dict[i.name] = i.get_data()
            return data_dict
        else:
            super().get_state()
