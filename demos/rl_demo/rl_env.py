from mojograsp.simcore.environment import Environment
from mojograsp.simobjects.two_finger_gripper import TwoFingerGripper
from mojograsp.simobjects.object_base import ObjectBase
import pybullet as p
import numpy as np
import time

class ExpertEnv(Environment):
    def __init__(self, hand: TwoFingerGripper, obj: ObjectBase, hand_type, physicsClientId=None,rand_start = False):
        self.hand = hand
        self.obj = obj
        if 'B' in hand_type:
            self.hand_type = 'B'
        else:
            self.hand_type = 'A'
        self.rand_start = rand_start
        
    def reset(self):
        # reset the simulator
        p.resetSimulation()
        # reload the objects
        plane_id = p.loadURDF("plane.urdf", flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
        
        #adding noise
        if self.rand_start:
            obj_change = np.random.normal(0,0.01,2)
        else:
            #no noise
            obj_change = np.array([0,0])
        
        

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
        mass_link = .036
        p.changeDynamics(hand_id, 1, lateralFriction=0.5, rollingFriction=0.04,
                         mass=.036)
        p.changeDynamics(hand_id, 4, lateralFriction=0.5, rollingFriction=0.04,
                         mass=.036)
        p.changeDynamics(hand_id, 0, jointLowerLimit=-1.57, jointUpperLimit=1.57, mass=mass_link)
        p.changeDynamics(hand_id, 1, jointLowerLimit=0, jointUpperLimit=2.09, mass=mass_link)
        p.changeDynamics(hand_id, 3, jointLowerLimit=-1.57, jointUpperLimit=1.57, mass=mass_link)
        p.changeDynamics(hand_id, 4, jointLowerLimit=-2.09, jointUpperLimit=0, mass=mass_link)
        
        obj_id = p.loadURDF(self.obj.path, basePosition=[0.0+obj_change[0], 0.10+obj_change[1], .05],
                        flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)

        if self.rand_start:
            f1_pos = [0.03+obj_change[0], 0.10+obj_change[1], 0.05]
            f2_pos = [-0.03+obj_change[0], 0.10+obj_change[1], 0.05]
            
            f1_angs = p.calculateInverseKinematics(hand_id, 2, f1_pos, maxNumIterations=3000)
            f2_angs = p.calculateInverseKinematics(hand_id, 5, f2_pos, maxNumIterations=3000)
            p.resetJointState(hand_id, 0, -np.pi/2)
            p.resetJointState(hand_id, 1, np.pi/4)
            p.resetJointState(hand_id, 3, np.pi/2)
            p.resetJointState(hand_id, 4, -np.pi/4)
            
            positions = np.linspace([-np.pi/2,np.pi/4,np.pi/2,-np.pi/4],[f1_angs[0],f1_angs[1],f2_angs[2],f2_angs[3]],20)
            for action_to_execute in positions:
                p.setJointMotorControlArray(hand_id, jointIndices=self.hand.get_joint_numbers(),
                                            controlMode=p.POSITION_CONTROL, targetPositions=action_to_execute,
                                            positionGains=[0.8,0.8,0.8,0.8], forces=[0.4,0.4,0.4,0.4])
                self.step()
                # time.sleep(0.01)
        # # print(f1_angs, f2_angs)
        # p.resetJointState(hand_id, 0, f1_angs[0])
        # p.resetJointState(hand_id, 1, f1_angs[1])
        # p.resetJointState(hand_id, 3, f2_angs[2])
        # p.resetJointState(hand_id, 4, f2_angs[3])
        p.changeDynamics(plane_id,-1,lateralFriction=0.5, spinningFriction=0.001, rollingFriction=0.0)
        p.changeDynamics(self.obj.id, -1, mass=.03, restitution=.95, lateralFriction=0.5, localInertiaDiagonal=[0.000029435425,0.000029435425,0.00000725805])
        # p.resetJointState(hand_id, 0, .695)
        # p.resetJointState(hand_id, 1, -1.487)
        # p.resetJointState(hand_id, 3, -.695)
        # p.resetJointState(hand_id, 4, 1.487)
        
        p.setGravity(0, 0, -10)
        p.setPhysicsEngineParameter(contactBreakingThreshold=.001)
        # obj_id = p.loadURDF(self.obj.path, basePosition=[0.0, 0.1067, .05])

        # Update the object id's
        self.hand.id = hand_id
        self.obj.id = obj_id
        # Change gripper color
        p.changeVisualShape(hand_id, -1, rgbaColor=[0.3, 0.3, 0.3, 1])
        p.changeVisualShape(hand_id, 0, rgbaColor=[1, 0.5, 0, 1])
        p.changeVisualShape(hand_id, 1, rgbaColor=[0.3, 0.3, 0.3, 1])
        p.changeVisualShape(hand_id, 3, rgbaColor=[1, 0.5, 0, 1])
        p.changeVisualShape(hand_id, 4, rgbaColor=[0.3, 0.3, 0.3, 1])
        p.changeVisualShape(obj_id, -1, rgbaColor=[0.1, 0.1, 0.1, 1])
        # time.sleep(5)

    def reset_to_pos(self, object_pos, finger_angles):
        # reset the simulator
        p.resetSimulation()
        # reload the objects
        plane_id = p.loadURDF("plane.urdf", flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)

        
        # For alt configuration
        hand_id = p.loadURDF(self.hand.path, useFixedBase=True,
                             basePosition=[0.0, 0.0, 0.05], flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)

        p.resetJointState(hand_id, 0, finger_angles[0])
        p.resetJointState(hand_id, 1, finger_angles[1])
        p.resetJointState(hand_id, 3, finger_angles[2])
        p.resetJointState(hand_id, 4, finger_angles[3])

        mass_link = .036
        p.changeDynamics(hand_id, 1, lateralFriction=0.5, rollingFriction=0.04,
                         mass=.036)
        p.changeDynamics(hand_id, 4, lateralFriction=0.5, rollingFriction=0.04,
                         mass=.036)
        p.changeDynamics(hand_id, 0, jointLowerLimit=-1.57, jointUpperLimit=1.57, mass=mass_link)
        p.changeDynamics(hand_id, 1, jointLowerLimit=0, jointUpperLimit=2.09, mass=mass_link)
        p.changeDynamics(hand_id, 3, jointLowerLimit=-1.57, jointUpperLimit=1.57, mass=mass_link)
        p.changeDynamics(hand_id, 4, jointLowerLimit=-2.09, jointUpperLimit=0, mass=mass_link)
        
        obj_id = p.loadURDF(self.obj.path, basePosition=[object_pos[0], object_pos[1], .05],
                        flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)


        p.changeDynamics(plane_id,-1,lateralFriction=0.5, spinningFriction=0.0, rollingFriction=0.0)
        p.changeDynamics(self.obj.id, -1, mass=.03, restitution=.95, lateralFriction=0.5)
        # p.resetJointState(hand_id, 0, .695)
        # p.resetJointState(hand_id, 1, -1.487)
        # p.resetJointState(hand_id, 3, -.695)
        # p.resetJointState(hand_id, 4, 1.487)
        
        p.setGravity(0, 0, -10)
        p.setPhysicsEngineParameter(contactBreakingThreshold=.001)
        # obj_id = p.loadURDF(self.obj.path, basePosition=[0.0, 0.1067, .05])

        # Update the object id's
        self.hand.id = hand_id
        self.obj.id = obj_id
        # Change gripper color
        p.changeVisualShape(hand_id, -1, rgbaColor=[0.3, 0.3, 0.3, 1])
        p.changeVisualShape(hand_id, 0, rgbaColor=[1, 0.5, 0, 1])
        p.changeVisualShape(hand_id, 1, rgbaColor=[0.3, 0.3, 0.3, 1])
        p.changeVisualShape(hand_id, 3, rgbaColor=[1, 0.5, 0, 1])
        p.changeVisualShape(hand_id, 4, rgbaColor=[0.3, 0.3, 0.3, 1])
        p.changeVisualShape(obj_id, -1, rgbaColor=[0.1, 0.1, 0.1, 1])
        # time.sleep(5)

    def setup(self):
        super().setup()

    def step(self):
        super().step()
