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
        pass

    @abstractmethod
    def get_reward(self) -> dict:
        """
        Method should return a dictionary that represents the resulting reward of an action
        AFTER it is executed and AFTER the sim is stepped. This could be distance measures, or any other 
        metrics saved in a dictionary

        :return: Dictionary containing the representation of the reward from an action
        :rtype: Dictionary with format {string: **ANY TYPE**}.
        """
        data_dict = {}
        return data_dict


class RewardDefault(Reward):
    """
    Default Reward Class that is used when the user does not need or wish to use the Reward class
    but it may still be called by other classes such as :func:`~mojograsp.simcore.record_data.RecordData` 
    and :func:`~mojograsp.simcore.replay_buffer.ReplayBufferDefault`.
    """

    def __init__(self):
        """Default Placeholder if no Reward class is provided"""
        super().__init__()

    def get_reward(self) -> dict:
        """Default method that returns an empty dictionary. 

        :return: An empty dictionary.
        :rtype: dict
        """
        super().get_reward()
