from abc import ABC, abstractmethod


class State(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_state(self) -> dict:
        pass


class StateDefault(State):
    # TODO Find reasonable default, best done after refactor of hand and object classes
    def __init__(self):
        super().__init__()

    def get_state(self) -> dict:
        super().get_state()
