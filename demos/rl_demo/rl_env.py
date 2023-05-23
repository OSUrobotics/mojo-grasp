from mojograsp.simcore.environment import Environment
from mojograsp.simobjects.two_finger_gripper import TwoFingerGripper
from mojograsp.simobjects.object_base import ObjectBase
import pybullet as p

class ExpertEnv(Environment):
    def __init__(self, hand: TwoFingerGripper, obj: ObjectBase, hand_type):
        self.hand = hand
        self.obj = obj
        if 'B' in hand_type:
            self.hand_type = 'B'
        else:
            self.hand_type = 'A'

    def reset(self):
        # reset the simulator
        p.resetSimulation()
        # reload the objects
        plane_id = p.loadURDF("plane.urdf")
        
        
        # hand_id = p.loadURDF(self.hand.path, useFixedBase=True,
        #                      basePosition=[0.0, 0.0, 0.05])
        # p.resetJointState(hand_id, 0, .75)
        # p.resetJointState(hand_id, 1, -1.4)
        # p.resetJointState(hand_id, 2, -.75)
        # p.resetJointState(hand_id, 3, 1.4)

        # For alt configuration
        hand_id = p.loadURDF(self.hand.path, useFixedBase=True,
                             basePosition=[0.0, 0.0, 0.05])
        # p.setPhysicsEngineParameter(contactBreakingThreshold = 0.001)
        # p.resetJointState(hand_id, 0, .75)
        # p.resetJointState(hand_id, 1, -1.4)
        # p.resetJointState(hand_id, 3, -.75)
        # p.resetJointState(hand_id, 4, 1.4)
        if self.hand_type =='A':
            p.resetJointState(hand_id, 0, -.725)
            p.resetJointState(hand_id, 1, 1.45)
            p.resetJointState(hand_id, 3, .725)
            p.resetJointState(hand_id, 4, -1.45)
        if self.hand_type == 'B':
            p.resetJointState(hand_id, 0, -.5)
            p.resetJointState(hand_id, 1, 1.5)
            p.resetJointState(hand_id, 3, .5)
            p.resetJointState(hand_id, 4, -1.5)
        p.changeDynamics(plane_id,-1,lateralFriction=0.05, spinningFriction=0.05, rollingFriction=0.05)
        
        # p.resetJointState(hand_id, 0, .695)
        # p.resetJointState(hand_id, 1, -1.487)
        # p.resetJointState(hand_id, 3, -.695)
        # p.resetJointState(hand_id, 4, 1.487)
        
        p.setGravity(0, 0, -10)
        
        # obj_id = p.loadURDF(self.obj.path, basePosition=[0.0, 0.1067, .05])

        obj_id = p.loadURDF(self.obj.path, basePosition=[0.0, 0.10, .05])
        # Update the object id's
        self.hand.id = hand_id
        self.obj.id = obj_id
        # Change gripper color
        p.changeVisualShape(hand_id, -1, rgbaColor=[0.3, 0.3, 0.3, 1])
        p.changeVisualShape(hand_id, 0, rgbaColor=[1, 0.5, 0, 1])
        p.changeVisualShape(hand_id, 1, rgbaColor=[0.3, 0.3, 0.3, 1])
        p.changeVisualShape(hand_id, 3, rgbaColor=[1, 0.5, 0, 1])
        p.changeVisualShape(hand_id, 4, rgbaColor=[0.3, 0.3, 0.3, 1])
        
    def setup(self):
        super().setup()

    def step(self):
        super().step()
