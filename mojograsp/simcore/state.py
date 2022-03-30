from abc import ABC, abstractmethod


class State(ABC):
    @abstractmethod
    def get_state(self) -> dict:
        pass


class StateBlank(State):
    def get_state(self) -> dict:
        return None


class StateDefault(State):
    # TODO Find reasonable default, best done after refactor of hand and object classes
    def __init__(self):
        pass

    def get_state(self) -> dict:
        pass
