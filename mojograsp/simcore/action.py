from abc import ABC, abstractmethod


class Action(ABC):
    """Action Abstract Base Class"""
    @abstractmethod
    def __init__(self):
        """
        Constructor should be used to pass any other objects or 
        data that an action will need to function. Usually this will be a manipulator 
        or other mojograsp objects.
        """
        pass

    @abstractmethod
    def get_action(self) -> dict:
        """
        Abstract method returns a dictionary that represents the intended target action
        BEFORE the sim is stepped. This could include joint target angles, 
        end effector targets, etc.

        :return: Dictionary containing the representation of an intended action for a simulator object.
        :rtype: Dictionary with format {string: **ANY TYPE**}.
        """
        data_dict = {}
        return data_dict


class ActionDefault(Action):
    """
    Default Action Class that is used when the user does not need or wish to use the Action class
    but it may still be called by other classes such as :func:`~mojograsp.simcore.record_data.RecordData` 
    and :func:`~mojograsp.simcore.replay_buffer.ReplayBufferDefault`.
    """

    def __init__(self):
        """Default constructor, takes no arguments.
        """
        super().__init__()

    def get_action(self) -> dict:
        """Default method that returns an empty dictionary. 

        :return: An empty dictionary.
        :rtype: dictionary.
        """
        super().get_action()
