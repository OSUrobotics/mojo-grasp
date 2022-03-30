from abc import ABC, abstractmethod


class Episode(ABC):
    sim = None

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def post_episode(self):
        pass


class EpisodeBlank(Episode):
    def setup(self):
        pass

    def reset(self):
        pass

    def post_episode(self):
        pass
