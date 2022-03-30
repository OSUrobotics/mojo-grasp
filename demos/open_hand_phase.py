import pybullet as p
from mojograsp.simcore.phase import Phase

from math import isclose


class OpenHand(Phase):

    def __init__(self):
        self.name = "open"
        self.target_pos = [1.57, 0, -1.57, 0]
        self.hand = self.env.hand
        self.object = self.env.object

    def setup(self):
        print("RUNNING 1")
        pass

    def execute_action(self):
        p.setJointMotorControlArray(self.hand.id, jointIndices=self.hand.actuation.get_joint_index_numbers(),
                                    controlMode=p.POSITION_CONTROL, targetPositions=self.target_pos)

    def exit_condition(self) -> bool:
        joint_index = self.hand.actuation.get_joint_index_numbers()
        joint_angles = self.hand.get_joint_angles(joint_index)

        for i in range(len(self.target_pos)):
            if not isclose(joint_angles[i], self.target_pos[i], abs_tol=1e-1):
                return False
        return True

    def next_phase(self) -> str:
        return "close"
