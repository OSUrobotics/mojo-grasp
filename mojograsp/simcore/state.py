from abc import ABC, abstractmethod

from mojograsp.simobjects.object_base import ObjectBase


class State(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_state(self) -> dict:
        data_dict = {}
        return data_dict


class StateDefault(State):
    # TODO Find reasonable default, best done after refactor of hand and object classes
    def __init__(self, objects: list = None):
        super().__init__()
        self.objects = objects

    def get_state(self) -> dict:
        if self.objects:
            data_dict = {}
            for i in self.objects:
                data_dict[i.name] = i.get_data()
            return data_dict
        else:
            super().get_state()
