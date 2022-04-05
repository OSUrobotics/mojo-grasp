from mojograsp.simcore.environment import Environment
from mojograsp.simobjects.hand import Hand
from mojograsp.simobjects.objectbase import ObjectBase
import pybullet as p


class AsteriskEnv(Environment):
    def __init__(self, hand: Hand, obj: ObjectBase):
        self.hand = hand
        self.obj = obj

    def reset(self):
        for i in self.hand.joint_dict.values():
            p.resetJointState(self.hand.id, i, 0)
        p.resetBasePositionAndOrientation(
            self.obj.id, [0.0, 0.17, .06], [0, 0, 0, 1])

    def setup(self):
        super().setup()

    def step(self):
        super().step()
