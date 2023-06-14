from mojograsp.simcore.environment import Environment
from mojograsp.simobjects.two_finger_gripper import TwoFingerGripper
from mojograsp.simobjects.object_base import ObjectBase
import pybullet as p
import numpy as np
import time

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
        plane_id = p.loadURDF("plane.urdf", flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
        
        #adding noise
        # obj_change = np.random.normal(0,0.01,2)
        #no noise
        obj_change = np.array([0,0])
        
        # f1_pos = [0.03+obj_change[0], 0.10+obj_change[1], 0.05]
        # f2_pos = [-0.03+obj_change[0], 0.10+obj_change[1], 0.05]
        # print('object pose', obj_change + np.array([0,0.1]))
        # print('f1 pos', f1_pos)
        # print('f2 pos', f2_pos)
        # hand_id = p.loadURDF(self.hand.path, useFixedBase=True,
        #                      basePosition=[0.0, 0.0, 0.05])
        # p.resetJointState(hand_id, 0, .75)
        # p.resetJointState(hand_id, 1, -1.4)
        # p.resetJointState(hand_id, 2, -.75)
        # p.resetJointState(hand_id, 3, 1.4)
        
        # For alt configuration
        hand_id = p.loadURDF(self.hand.path, useFixedBase=True,
                             basePosition=[0.0, 0.0, 0.05], flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
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
        # f1_angs = p.calculateInverseKinematics(hand_id, 2, f1_pos, maxNumIterations=3000)
        # f2_angs = p.calculateInverseKinematics(hand_id, 5, f2_pos, maxNumIterations=3000)
        # # print(f1_angs, f2_angs)
        # p.resetJointState(hand_id, 0, f1_angs[0])
        # p.resetJointState(hand_id, 1, f1_angs[1])
        # p.resetJointState(hand_id, 3, f2_angs[2])
        # p.resetJointState(hand_id, 4, f2_angs[3])
        p.changeDynamics(plane_id,-1,lateralFriction=0.05, spinningFriction=0.05, rollingFriction=0.05)
        
        # p.resetJointState(hand_id, 0, .695)
        # p.resetJointState(hand_id, 1, -1.487)
        # p.resetJointState(hand_id, 3, -.695)
        # p.resetJointState(hand_id, 4, 1.487)
        
        p.setGravity(0, 0, -10)
        
        # obj_id = p.loadURDF(self.obj.path, basePosition=[0.0, 0.1067, .05])

        obj_id = p.loadURDF(self.obj.path, basePosition=[0.0+obj_change[0], 0.10+obj_change[1], .05], flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
        # Update the object id's
        self.hand.id = hand_id
        self.obj.id = obj_id
        # Change gripper color
        p.changeVisualShape(hand_id, -1, rgbaColor=[0.3, 0.3, 0.3, 1])
        p.changeVisualShape(hand_id, 0, rgbaColor=[1, 0.5, 0, 1])
        p.changeVisualShape(hand_id, 1, rgbaColor=[0.3, 0.3, 0.3, 1])
        p.changeVisualShape(hand_id, 3, rgbaColor=[1, 0.5, 0, 1])
        p.changeVisualShape(hand_id, 4, rgbaColor=[0.3, 0.3, 0.3, 1])
        # time.sleep(5)
        
    def setup(self):
        super().setup()

    def step(self):
        super().step()
