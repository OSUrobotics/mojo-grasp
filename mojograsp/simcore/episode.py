from abc import ABC, abstractmethod


class Episode(ABC):
    """Episode Abstract Base Class"""

    @abstractmethod
    def __init__(self):
        """
        Constructor should be used to pass any other objects or 
        data that the Episode class may need to setup pre episode or post episode.
        Usually this will be a manipulator or other mojograsp objects.
        """
        pass

    @abstractmethod
    def setup(self):
        """
        This method is called by :func:`~mojograsp.simcore.sim_manager.SimManager` once before an episode begins.
        """
        pass

    @abstractmethod
    def post_episode(self):
        """
        This method is called by :func:`~mojograsp.simcore.sim_manager.SimManager` once after an episode ends.
        """
        pass


class EpisodeDefault(Episode):
    def __init__(self):
        """Default Placeholder if no Episode class is provided"""
        super().__init__()

    def setup(self):
        """Default Placeholder if no Episode class is provided"""
        super().setup()

    def post_episode(self):
        """Default Placeholder if no Episode class is provided"""
        super().post_episode()
