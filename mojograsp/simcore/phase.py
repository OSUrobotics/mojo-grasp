from abc import ABC, abstractmethod


class Phase(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def execute_action(self):
        pass

    @abstractmethod
    def exit_condition(self) -> bool:
        pass

    @abstractmethod
    def next_phase(self) -> str:
        pass

    def pre_step(self):
        pass

    def post_step(self):
        pass
