import pybullet as p
from mojograsp.simcore.phase import Phase
from mojograsp.simobjects.two_finger_gripper import TwoFingerGripper
from mojograsp.simobjects.object_base import ObjectBase

from math import isclose


class CloseHand(Phase):

    def __init__(self, hand: TwoFingerGripper, obj: ObjectBase):
        self.name = "close"
        self.target_pos = [0.7, -1.32, -0.7, 1.32]
        self.hand = hand
        self.obj = obj

    def setup(self):
        roll_fric = 0.01
        # object
        p.changeDynamics(self.obj.id, -1, mass=0.04, rollingFriction=roll_fric)
        # distal
        p.changeDynamics(self.hand.id, 1, mass=0.03, rollingFriction=roll_fric)
        p.changeDynamics(self.hand.id, 3, mass=0.03, rollingFriction=roll_fric)
        # proximal
        p.changeDynamics(self.hand.id, 0, mass=0.02, rollingFriction=roll_fric)
        p.changeDynamics(self.hand.id, 2, mass=0.02, rollingFriction=roll_fric)

    def execute_action(self):
        p.setJointMotorControlArray(self.hand.id, jointIndices=self.hand.get_joint_numbers(),
                                    controlMode=p.POSITION_CONTROL, targetPositions=self.target_pos)

    def exit_condition(self) -> bool:
        joint_index = self.hand.get_joint_numbers()
        joint_angles = self.hand.get_joint_angles(joint_index)

        for i in range(len(self.target_pos)):
            if not isclose(joint_angles[i], self.target_pos[i], abs_tol=1e-1):
                return False
        return True

    def next_phase(self) -> str:
        return None
