import pybullet as p
from mojograsp.simcore.phase import Phase
from mojograsp.simobjects.hand import Hand
from mojograsp.simobjects.objectbase import ObjectBase

from math import isclose


class OpenHand(Phase):

    def __init__(self, hand: Hand, obj: ObjectBase):
        self.name = "open"
        self.target_pos = [1.57, 0, -1.57, 0]
        self.hand = hand
        self.obj = obj

    def setup(self):
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
