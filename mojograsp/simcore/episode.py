from abc import ABC, abstractmethod


class Episode(ABC):
    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def post_episode(self):
        pass


class EpisodeDefault(Episode):
    def setup(self):
        super().setup()

    def post_episode(self):
        super().post_episode()
