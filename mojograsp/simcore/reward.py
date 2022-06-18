from abc import ABC, abstractmethod


class Reward(ABC):
    """Reward Abstract Base Class"""
    @abstractmethod
    def __init__(self):
        """
        Constructor should be used to pass any other objects or 
        data that the reward will need to function. Usually this will be a manipulator 
        or other mojograsp objects.
        """
        self.current_reward = {}
        pass

    @abstractmethod
    def set_reward(self):
        """
        Method should be used to set the class variable self.current_reward. It should be called in the post_step()
        method of a user defined phase. This can be set however you would like, as long as self.current_reward
        is updated so that get_reward() is returns properly. This could be distance measures, or any other 
        metrics saved in a dictionary
        """
        self.current_reward = {}

    @abstractmethod
    def get_reward(self) -> dict:
        """
        Method should return a dictionary that represents the resulting reward of an action
        AFTER it is executed and AFTER the sim is stepped. 

        :return: Dictionary containing the representation of the reward from an action
        :rtype: Dictionary with format {string: **ANY TYPE**}.
        """
        return self.current_reward


class RewardDefault(Reward):
    """
    Default Reward Class that is used when the user does not need or wish to use the Reward class
    but it may still be called by other classes such as :func:`~mojograsp.simcore.record_data.RecordData` 
    and :func:`~mojograsp.simcore.replay_buffer.ReplayBufferDefault`.
    """

    def __init__(self):
        """Default Placeholder if no Reward class is provided"""
        super().__init__()

    def set_reward(self):
        """
        Default method that sets self.current_reward to an empty dictionary. 
        """
        super().set_reward()

    def get_reward(self) -> dict:
        """Default method that returns an empty dictionary. 

        :return: An empty dictionary.
        :rtype: dict
        """
        super().get_reward()
