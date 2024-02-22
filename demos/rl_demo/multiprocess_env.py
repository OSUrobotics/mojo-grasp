from mojograsp.simcore.environment import Environment
from mojograsp.simobjects.two_finger_gripper import TwoFingerGripper
from mojograsp.simobjects.object_base import ObjectBase
import numpy as np
import time
import os

class MultiprocessEnv():
    def __init__(self, pybullet_import, hand: TwoFingerGripper, obj: ObjectBase, hand_type, physicsClientId=None,rand_start = False):
        self.hand = hand
        self.obj = obj
        if 'B' in hand_type:
            self.hand_type = 'B'
        else:
            self.hand_type = 'A'
        self.rand_start = rand_start
        self.p=pybullet_import
        
    def reset(self):
        # reset the simulator
        self.p.resetSimulation()
        # reload the objects
        plane_id = self.p.loadURDF("plane.urdf", flags=self.p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
        
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
        hand_id = self.p.loadURDF(self.hand.path, useFixedBase=True,
                             basePosition=[0.0, 0.0, 0.05], flags=self.p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
        # p.setPhysicsEngineParameter(contactBreakingThreshold = 0.001)
        # p.resetJointState(hand_id, 0, .75)
        # p.resetJointState(hand_id, 1, -1.4)
        # p.resetJointState(hand_id, 3, -.75)
        # p.resetJointState(hand_id, 4, 1.4)

        if self.hand_type =='A':
            self.p.resetJointState(hand_id, 0, -.725)
            self.p.resetJointState(hand_id, 1, 1.45)
            self.p.resetJointState(hand_id, 3, .725)
            self.p.resetJointState(hand_id, 4, -1.45)
        if self.hand_type == 'B':
            
            self.p.resetJointState(hand_id, 0, -.46)
            self.p.resetJointState(hand_id, 1, 1.5)
            self.p.resetJointState(hand_id, 3, .46)
            self.p.resetJointState(hand_id, 4, -1.5)
        mass_link = .036
        self.p.changeDynamics(hand_id, 1, lateralFriction=0.5, rollingFriction=0.04,
                         mass=.036)
        self.p.changeDynamics(hand_id, 4, lateralFriction=0.5, rollingFriction=0.04,
                         mass=.036)
        self.p.changeDynamics(hand_id, 0, jointLowerLimit=-1.57, jointUpperLimit=1.57, mass=mass_link)
        self.p.changeDynamics(hand_id, 1, jointLowerLimit=0, jointUpperLimit=2.09, mass=mass_link)
        self.p.changeDynamics(hand_id, 3, jointLowerLimit=-1.57, jointUpperLimit=1.57, mass=mass_link)
        self.p.changeDynamics(hand_id, 4, jointLowerLimit=-2.09, jointUpperLimit=0, mass=mass_link)
        
        obj_id = self.p.loadURDF(self.obj.path, basePosition=[0.0+obj_change[0], 0.10+obj_change[1], .05],
                        flags=self.p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)

        if self.rand_start:
            f1_pos = [0.03+obj_change[0], 0.10+obj_change[1], 0.05]
            f2_pos = [-0.03+obj_change[0], 0.10+obj_change[1], 0.05]
            
            f1_angs = self.p.calculateInverseKinematics(hand_id, 2, f1_pos, maxNumIterations=3000)
            f2_angs = self.p.calculateInverseKinematics(hand_id, 5, f2_pos, maxNumIterations=3000)
            self.p.resetJointState(hand_id, 0, -np.pi/2)
            self.p.resetJointState(hand_id, 1, np.pi/4)
            self.p.resetJointState(hand_id, 3, np.pi/2)
            self.p.resetJointState(hand_id, 4, -np.pi/4)
            
            positions = np.linspace([-np.pi/2,np.pi/4,np.pi/2,-np.pi/4],[f1_angs[0],f1_angs[1],f2_angs[2],f2_angs[3]],20)
            for action_to_execute in positions:
                self.p.setJointMotorControlArray(hand_id, jointIndices=self.hand.get_joint_numbers(),
                                            controlMode=self.p.POSITION_CONTROL, targetPositions=action_to_execute,
                                            positionGains=[0.8,0.8,0.8,0.8], forces=[0.4,0.4,0.4,0.4])
                self.step()
                # time.sleep(0.01)
        # # print(f1_angs, f2_angs)
        # p.resetJointState(hand_id, 0, f1_angs[0])
        # p.resetJointState(hand_id, 1, f1_angs[1])
        # p.resetJointState(hand_id, 3, f2_angs[2])
        # p.resetJointState(hand_id, 4, f2_angs[3])
        self.p.changeDynamics(plane_id,-1,lateralFriction=0.05, spinningFriction=0.05, rollingFriction=0.05)
        self.p.changeDynamics(self.obj.id, -1, mass=.03, restitution=.95, lateralFriction=0.5)
        # p.resetJointState(hand_id, 0, .695)
        # p.resetJointState(hand_id, 1, -1.487)
        # p.resetJointState(hand_id, 3, -.695)
        # p.resetJointState(hand_id, 4, 1.487)
        
        self.p.setGravity(0, 0, -10)
        self.p.setPhysicsEngineParameter(contactBreakingThreshold=.001)
        # obj_id = p.loadURDF(self.obj.path, basePosition=[0.0, 0.1067, .05])

        # Update the object id's
        self.hand.id = hand_id
        self.obj.id = obj_id
        # Change gripper color
        self.p.changeVisualShape(hand_id, -1, rgbaColor=[0.3, 0.3, 0.3, 1])
        self.p.changeVisualShape(hand_id, 0, rgbaColor=[1, 0.5, 0, 1])
        self.p.changeVisualShape(hand_id, 1, rgbaColor=[0.3, 0.3, 0.3, 1])
        self.p.changeVisualShape(hand_id, 3, rgbaColor=[1, 0.5, 0, 1])
        self.p.changeVisualShape(hand_id, 4, rgbaColor=[0.3, 0.3, 0.3, 1])
        self.p.changeVisualShape(obj_id, -1, rgbaColor=[0.1, 0.1, 0.1, 1])
        # time.sleep(5)

    def reset_to_pos(self, object_pos, finger_angles):
        # reset the simulator
        self.p.resetSimulation()
        # reload the objects
        plane_id = self.p.loadURDF("plane.urdf", flags=self.p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)

        
        # For alt configuration
        hand_id = self.p.loadURDF(self.hand.path, useFixedBase=True,
                             basePosition=[0.0, 0.0, 0.05], flags=self.p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)

        self.p.resetJointState(hand_id, 0, finger_angles[0])
        self.p.resetJointState(hand_id, 1, finger_angles[1])
        self.p.resetJointState(hand_id, 3, finger_angles[2])
        self.p.resetJointState(hand_id, 4, finger_angles[3])

        mass_link = .036
        self.p.changeDynamics(hand_id, 1, lateralFriction=0.5, rollingFriction=0.04,
                         mass=.036)
        self.p.changeDynamics(hand_id, 4, lateralFriction=0.5, rollingFriction=0.04,
                         mass=.036)
        self.p.changeDynamics(hand_id, 0, jointLowerLimit=-1.57, jointUpperLimit=1.57, mass=mass_link)
        self.p.changeDynamics(hand_id, 1, jointLowerLimit=0, jointUpperLimit=2.09, mass=mass_link)
        self.p.changeDynamics(hand_id, 3, jointLowerLimit=-1.57, jointUpperLimit=1.57, mass=mass_link)
        self.p.changeDynamics(hand_id, 4, jointLowerLimit=-2.09, jointUpperLimit=0, mass=mass_link)
        
        obj_id = self.p.loadURDF(self.obj.path, basePosition=[object_pos[0], object_pos[1], .05],
                        flags=self.p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)


        self.p.changeDynamics(plane_id,-1,lateralFriction=0.5, spinningFriction=0.0, rollingFriction=0.0)
        self.p.changeDynamics(self.obj.id, -1, mass=.03, restitution=.95, lateralFriction=0.5)
        # p.resetJointState(hand_id, 0, .695)
        # p.resetJointState(hand_id, 1, -1.487)
        # p.resetJointState(hand_id, 3, -.695)
        # p.resetJointState(hand_id, 4, 1.487)
        
        self.p.setGravity(0, 0, -10)
        self.p.setPhysicsEngineParameter(contactBreakingThreshold=.001)
        # obj_id = p.loadURDF(self.obj.path, basePosition=[0.0, 0.1067, .05])

        # Update the object id's
        self.hand.id = hand_id
        self.obj.id = obj_id
        # Change gripper color
        self.p.changeVisualShape(hand_id, -1, rgbaColor=[0.3, 0.3, 0.3, 1])
        self.p.changeVisualShape(hand_id, 0, rgbaColor=[1, 0.5, 0, 1])
        self.p.changeVisualShape(hand_id, 1, rgbaColor=[0.3, 0.3, 0.3, 1])
        self.p.changeVisualShape(hand_id, 3, rgbaColor=[1, 0.5, 0, 1])
        self.p.changeVisualShape(hand_id, 4, rgbaColor=[0.3, 0.3, 0.3, 1])
        self.p.changeVisualShape(obj_id, -1, rgbaColor=[0.1, 0.1, 0.1, 1])
        # time.sleep(5)

    def setup(self):
        pass

    def step(self):
        self.p.stepSimulation()

class MultiprocessSingleShapeEnv(Environment):
    def __init__(self,pybulletInstance, hand: TwoFingerGripper, obj: ObjectBase, hand_type ,rand_start = 'N'):
        self.hand = hand
        self.obj = obj
        self.p = pybulletInstance
        mass_link = .036
        key_nums = hand_type.split('_')
        self.f1 = key_nums[1]
        self.f2 = key_nums[2]
        self.width = key_nums[-1]
            
        if rand_start =='obj':
            self.rand_start = True
            self.rand_finger_pos = False
            self.end_start = False
        elif rand_start =='finger':
            self.rand_finger_pos = True
            self.rand_start = False
            self.end_start = False
        elif rand_start =='end':
            self.end_start = True
            self.rand_start = False
            self.rand_finger_pos = False
        else:
            self.rand_finger_pos = False
            self.rand_start = False
            self.end_start = False

        
        self.p.resetSimulation()
        self.plane_id = self.p.loadURDF("plane.urdf", flags=self.p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
        self.hand_id = self.p.loadURDF(self.hand.path, useFixedBase=True,
                             basePosition=[0.0, 0.0, 0.05], flags=self.p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
        self.obj_id = self.p.loadURDF(self.obj.path, basePosition=[0.0, 0.10, .05],
                                 flags=self.p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
        self.p.changeDynamics(self.hand_id, 1, lateralFriction=0.5, rollingFriction=0.04,
                         mass=.036)
        self.p.changeDynamics(self.hand_id, 4, lateralFriction=0.5, rollingFriction=0.04,
                         mass=.036)
        self.p.changeDynamics(self.hand_id, 0, jointLowerLimit=-1.57, jointUpperLimit=1.57, mass=mass_link)
        self.p.changeDynamics(self.hand_id, 1, jointLowerLimit=0, jointUpperLimit=2.09, mass=mass_link)
        self.p.changeDynamics(self.hand_id, 3, jointLowerLimit=-1.57, jointUpperLimit=1.57, mass=mass_link)
        self.p.changeDynamics(self.hand_id, 4, jointLowerLimit=-2.09, jointUpperLimit=0, mass=mass_link)
        self.p.changeDynamics(self.plane_id,-1,lateralFriction=0.5, spinningFriction=0.001, rollingFriction=0.0)
        self.p.changeDynamics(self.obj.id, -1, mass=.03, restitution=.95, lateralFriction=0.5, localInertiaDiagonal=[0.000029435425,0.000029435425,0.00000725805])
        self.p.changeVisualShape(self.hand_id, -1, rgbaColor=[0.3, 0.3, 0.3, 1])
        self.p.changeVisualShape(self.hand_id, 0, rgbaColor=[1, 0.5, 0, 1])
        self.p.changeVisualShape(self.hand_id, 1, rgbaColor=[0.3, 0.3, 0.3, 1])
        self.p.changeVisualShape(self.hand_id, 3, rgbaColor=[1, 0.5, 0, 1])
        self.p.changeVisualShape(self.hand_id, 4, rgbaColor=[0.3, 0.3, 0.3, 1])
        self.p.changeVisualShape(self.obj_id, -1, rgbaColor=[0.1, 0.1, 0.1, 1])
        self.hand.id = self.hand_id
        self.obj.id = self.obj_id
        self.start_time = 0
        
        self.p.setGravity(0, 0, -10)
        self.p.setPhysicsEngineParameter(contactBreakingThreshold=.001)

    def reset(self, start_pos=None):
        # reset the simulator
        if start_pos is not None:
            obj_change = start_pos
        elif self.rand_start:
            obj_change = np.random.normal(0,0.01,2)
        else:
            #no noise
            obj_change = np.array([0,0])


        self.p.resetJointState(self.hand.id, 0, self.hand.starting_angles[0])
        self.p.resetJointState(self.hand.id, 1, self.hand.starting_angles[1])
        self.p.resetJointState(self.hand.id, 3, self.hand.starting_angles[2])
        self.p.resetJointState(self.hand.id, 4, self.hand.starting_angles[3])
        
        self.p.resetBasePositionAndOrientation(self.obj_id, posObj=[0.0+obj_change[0], 0.10+obj_change[1], .05], ornObj=[0,0,0,1])
        f1_dist = self.p.getClosestPoints(self.obj.id, self.hand.id, 10, -1, 1, -1)
        f2_dist = self.p.getClosestPoints(self.obj.id, self.hand.id, 10, -1, 4, -1)
        # print('f1 dist', f1_dist)
        # print('f2_dist', f2_dist)
        # print('finger pos', self.p.getLinkState(self.hand.id, 2)[0], self.p.getLinkState(self.hand.id, 5)[0])
        # print('joint info', self.p.getJointInfo(self.hand.id,0), self.p.getJointInfo(self.hand.id,1),self.p.getJointInfo(self.hand.id,3),self.p.getJointInfo(self.hand.id,4))

        if self.rand_start:
            # print('are we here?')
            f1_pos = [0.03+obj_change[0], 0.10+obj_change[1], 0.05]
            f2_pos = [-0.03+obj_change[0], 0.10+obj_change[1], 0.05]
            
            f1_angs = self.p.calculateInverseKinematics(self.hand_id, 2, f1_pos, maxNumIterations=3000)
            f2_angs = self.p.calculateInverseKinematics(self.hand_id, 5, f2_pos, maxNumIterations=3000)
            self.p.resetJointState(self.hand_id, 0, -np.pi/2)
            self.p.resetJointState(self.hand_id, 1, np.pi/4)
            self.p.resetJointState(self.hand_id, 3, np.pi/2)
            self.p.resetJointState(self.hand_id, 4, -np.pi/4)
            
            positions = np.linspace([-np.pi/2,np.pi/4,np.pi/2,-np.pi/4],[f1_angs[0],f1_angs[1],f2_angs[2],f2_angs[3]],20)
            for action_to_execute in positions:
                self.p.setJointMotorControlArray(self.hand_id, jointIndices=self.hand.get_joint_numbers(),
                                            controlMode=self.p.POSITION_CONTROL, targetPositions=action_to_execute,
                                            positionGains=[0.8,0.8,0.8,0.8], forces=[0.4,0.4,0.4,0.4])
                self.step()
        if self.end_start:
            link1_pose = self.p.getLinkState(self.hand_id, 2)[0]
            link2_pose = self.p.getLinkState(self.hand_id, 5)[0]
            f1_pos = [link1_pose[0]+obj_change[0], link1_pose[1] + obj_change[1], 0.05]
            f2_pos = [link2_pose[0]+obj_change[0], link2_pose[1] + obj_change[1], 0.05]
            
            f1_angs = self.p.calculateInverseKinematics(self.hand_id, 2, f1_pos, maxNumIterations=3000)
            f2_angs = self.p.calculateInverseKinematics(self.hand_id, 5, f2_pos, maxNumIterations=3000)
            self.p.resetJointState(self.hand_id, 0, -np.pi/2)
            self.p.resetJointState(self.hand_id, 1, np.pi/4)
            self.p.resetJointState(self.hand_id, 3, np.pi/2)
            self.p.resetJointState(self.hand_id, 4, -np.pi/4)
            
            positions = np.linspace([-np.pi/2,np.pi/4,np.pi/2,-np.pi/4],[f1_angs[0],f1_angs[1],f2_angs[2],f2_angs[3]],20)
            for action_to_execute in positions:
                self.p.setJointMotorControlArray(self.hand_id, jointIndices=self.hand.get_joint_numbers(),
                                            controlMode=self.p.POSITION_CONTROL, targetPositions=action_to_execute,
                                            positionGains=[0.8,0.8,0.8,0.8], forces=[0.4,0.4,0.4,0.4])
                self.step()
        if self.rand_finger_pos:
            # print('we here')
            y_change = np.random.uniform(-0.01,0.01,2)
            # print(y_change)
            link1_pose = self.p.getLinkState(self.hand_id, 2)[0]
            link2_pose = self.p.getLinkState(self.hand_id, 5)[0]
            # link1_pose = [0.026749999999999996, 0.10778391676312778, 0.05]
            # link2_pose = [-0.026749999999999996, 0.10778391676312778, 0.05]
            # print('starting link poses',link1_pose,link2_pose)
            f1_pos = [link1_pose[0], link1_pose[1] + y_change[0], 0.05]
            f2_pos = [link2_pose[0], link2_pose[1] + y_change[1], 0.05]
            f1_angs = self.p.calculateInverseKinematics(self.hand_id, 2, f1_pos, maxNumIterations=3000)
            f2_angs = self.p.calculateInverseKinematics(self.hand_id, 5, f2_pos, maxNumIterations=3000)
            self.p.resetJointState(self.hand_id, 0, -np.pi/2)
            self.p.resetJointState(self.hand_id, 1, np.pi/4)
            self.p.resetJointState(self.hand_id, 3, np.pi/2)
            self.p.resetJointState(self.hand_id, 4, -np.pi/4)
            
            positions = np.linspace([-np.pi/2,np.pi/4,np.pi/2,-np.pi/4],[f1_angs[0],f1_angs[1],f2_angs[2],f2_angs[3]],20)
            for action_to_execute in positions:
                self.p.setJointMotorControlArray(self.hand_id, jointIndices=self.hand.get_joint_numbers(),
                                            controlMode=self.p.POSITION_CONTROL, targetPositions=action_to_execute,
                                            positionGains=[0.8,0.8,0.8,0.8], forces=[0.4,0.4,0.4,0.4])
                self.step()
            # print('positions', f1_angs,f2_angs)
            # print('fuck')
            # print('finger tip poses', f1_pos, f2_pos)
            # f1_dist = self.p.getClosestPoints(self.obj.id, self.hand.id, 10, -1, 1, -1)
            # f2_dist = self.p.getClosestPoints(self.obj.id, self.hand.id, 10, -1, 4, -1)
            # obj_velocity = self.p.getBaseVelocity(self.obj.id)
            # print('f1 dist', f1_dist[0][8])
            # print('f2_dist', f2_dist[0][8])
            # print(obj_velocity)
        # input('fak')

    def reset_to_pos(self, object_pos, finger_angles):
        # reset the simulator
        self.p.resetSimulation()
        # reload the objects
        plane_id = self.p.loadURDF("plane.urdf", flags=self.p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)

        
        # For alt configuration
        hand_id = self.p.loadURDF(self.hand.path, useFixedBase=True,
                             basePosition=[0.0, 0.0, 0.05], flags=self.p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)

        self.p.resetJointState(hand_id, 0, finger_angles[0])
        self.p.resetJointState(hand_id, 1, finger_angles[1])
        self.p.resetJointState(hand_id, 3, finger_angles[2])
        self.p.resetJointState(hand_id, 4, finger_angles[3])

        mass_link = .036
        self.p.changeDynamics(hand_id, 1, lateralFriction=0.5, rollingFriction=0.04,
                         mass=.036)
        self.p.changeDynamics(hand_id, 4, lateralFriction=0.5, rollingFriction=0.04,
                         mass=.036)
        self.p.changeDynamics(hand_id, 0, jointLowerLimit=-1.57, jointUpperLimit=1.57, mass=mass_link)
        self.p.changeDynamics(hand_id, 1, jointLowerLimit=0, jointUpperLimit=2.09, mass=mass_link)
        self.p.changeDynamics(hand_id, 3, jointLowerLimit=-1.57, jointUpperLimit=1.57, mass=mass_link)
        self.p.changeDynamics(hand_id, 4, jointLowerLimit=-2.09, jointUpperLimit=0, mass=mass_link)
        
        obj_id = self.p.loadURDF(self.obj.path, basePosition=[object_pos[0], object_pos[1], .05],
                        flags=self.p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)


        self.p.changeDynamics(plane_id,-1,lateralFriction=0.5, spinningFriction=0.0, rollingFriction=0.0)
        self.p.changeDynamics(self.obj.id, -1, mass=.03, restitution=.95, lateralFriction=0.5)
        
        self.p.setGravity(0, 0, -10)
        self.p.setPhysicsEngineParameter(contactBreakingThreshold=.001)
        # obj_id = self.p.loadURDF(self.obj.path, basePosition=[0.0, 0.1067, .05])

        # Update the object id's
        self.hand.id = hand_id
        self.obj.id = obj_id
        # Change gripper color
        self.p.changeVisualShape(hand_id, -1, rgbaColor=[0.3, 0.3, 0.3, 1])
        self.p.changeVisualShape(hand_id, 0, rgbaColor=[1, 0.5, 0, 1])
        self.p.changeVisualShape(hand_id, 1, rgbaColor=[0.3, 0.3, 0.3, 1])
        self.p.changeVisualShape(hand_id, 3, rgbaColor=[1, 0.5, 0, 1])
        self.p.changeVisualShape(hand_id, 4, rgbaColor=[0.3, 0.3, 0.3, 1])
        self.p.changeVisualShape(obj_id, -1, rgbaColor=[0.1, 0.1, 0.1, 1])
        # time.sleep(5)

    def setup(self):
        super().setup()

    def step(self):
        super().step()
        