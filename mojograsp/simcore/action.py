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
        self.current_action = {}
        pass

    @abstractmethod
    def set_action(self):
        """
        Method should be used to set the class variable self.current_action. It should be called in the pre_step()
        method of a user defined phase. This can be set however you would like, as long as self.current_action
        is updated so that get_action is returns properly. This could include joint target angles, 
        end effector targets, etc.

        """
        self.current_action = {}

    @abstractmethod
    def get_action(self) -> dict:
        """
        Method should return a dictionary that represents the intended target action
        BEFORE it is executed and the sim is stepped. The return value is class variable 
        self.current_action which is updated by set_action(). 

        :return: Dictionary containing the representation of an intended action for a simulator object.
        :rtype: Dictionary with format {string: **ANY TYPE**}.
        """
        return self.current_action


class ActionDefault(Action):
    """
    Default Action Class that is used when the user does not need or wish to use the Action class
    but it may still be called by other classes such as :func:`~mojograsp.simcore.record_data.RecordData` 
    and :func:`~mojograsp.simcore.replay_buffer.ReplayBufferDefault`.
    """

    def __init__(self):
        """Default Placeholder if no Action class is provided"""
        super().__init__()

    def set_action(self):
        """
        Default method that sets self.current_action to an empty dictionary. 
        """
        super().set_action()

    def get_action(self) -> dict:
        """
        Default method that returns an empty dictionary. 

        :return: An empty dictionary.
        :rtype: dict
        """
        super().get_action()
