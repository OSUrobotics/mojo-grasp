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
        self.start_time = 0
        
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

        
        # For alt configuration
        hand_id = p.loadURDF(self.hand.path, useFixedBase=True,
                             basePosition=[0.0, 0.0, 0.05], flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)

        if self.hand_type =='A':
            p.resetJointState(hand_id, 0, -.725)
            p.resetJointState(hand_id, 1, 1.45)
            p.resetJointState(hand_id, 3, .725)
            p.resetJointState(hand_id, 4, -1.45)
        if self.hand_type == 'B':
            p.resetJointState(self.hand_id, 0, -.46)
            p.resetJointState(self.hand_id, 1, 1.5)
            p.resetJointState(self.hand_id, 3, .46)
            p.resetJointState(self.hand_id, 4, -1.5)
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

        p.changeDynamics(plane_id,-1,lateralFriction=0.5, spinningFriction=0.001, rollingFriction=0.0)
        p.changeDynamics(self.obj.id, -1, mass=.03, restitution=.95, lateralFriction=0.5, localInertiaDiagonal=[0.000029435425,0.000029435425,0.00000725805])

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

class SingleShapeEnv(Environment):
    def __init__(self, hand: TwoFingerGripper, obj: ObjectBase, hand_type, physicsClientId=None,rand_start = 'no'):
        self.hand = hand
        self.obj = obj
        mass_link = .036
        key_nums = hand_type.split('_')
        self.f1 = key_nums[1]
        self.f2 = key_nums[2]
        self.width = key_nums[-1]
        np.random.seed(42)
        if rand_start =='obj':
            self.rand_start = True
            self.rand_finger_pos = False
        elif rand_start =='finger':
            self.rand_finger_pos = True
            self.rand_start = False
        else:
            self.rand_finger_pos = False
            self.rand_start = False
        p.resetSimulation()
        self.plane_id = p.loadURDF("plane.urdf", flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
        self.hand_id = p.loadURDF(self.hand.path, useFixedBase=True,
                             basePosition=[0.0, 0.0, 0.05], flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
        self.obj_id = p.loadURDF(self.obj.path, basePosition=[0.0, 0.10, .05],
                                 flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
        p.changeDynamics(self.hand_id, 1, lateralFriction=0.5, rollingFriction=0.04,
                         mass=.036)
        p.changeDynamics(self.hand_id, 4, lateralFriction=0.5, rollingFriction=0.04,
                         mass=.036)
        p.changeDynamics(self.hand_id, 0, jointLowerLimit=-1.57, jointUpperLimit=1.57, mass=mass_link)
        p.changeDynamics(self.hand_id, 1, jointLowerLimit=0, jointUpperLimit=2.09, mass=mass_link)
        p.changeDynamics(self.hand_id, 3, jointLowerLimit=-1.57, jointUpperLimit=1.57, mass=mass_link)
        p.changeDynamics(self.hand_id, 4, jointLowerLimit=-2.09, jointUpperLimit=0, mass=mass_link)
        p.changeDynamics(self.plane_id,-1,lateralFriction=0.5, spinningFriction=0.001, rollingFriction=0.0)
        p.changeDynamics(self.obj.id, -1, mass=.03, restitution=.95, lateralFriction=0.5, localInertiaDiagonal=[0.000029435425,0.000029435425,0.00000725805])
        p.changeVisualShape(self.hand_id, -1, rgbaColor=[0.3, 0.3, 0.3, 1])
        p.changeVisualShape(self.hand_id, 0, rgbaColor=[1, 0.5, 0, 1])
        p.changeVisualShape(self.hand_id, 1, rgbaColor=[0.3, 0.3, 0.3, 1])
        p.changeVisualShape(self.hand_id, 3, rgbaColor=[1, 0.5, 0, 1])
        p.changeVisualShape(self.hand_id, 4, rgbaColor=[0.3, 0.3, 0.3, 1])
        p.changeVisualShape(self.obj_id, -1, rgbaColor=[0.1, 0.1, 0.1, 1])
        self.hand.id = self.hand_id
        self.obj.id = self.obj_id
        self.start_time = 0
        
        p.setGravity(0, 0, -10)
        p.setPhysicsEngineParameter(contactBreakingThreshold=.001)

    def reset(self):
        # reset the simulator
        if self.rand_start:
            obj_change = np.random.normal(0,0.01,2)
        else:
            #no noise
            obj_change = np.array([0,0])

        if self.width == '53':
            if self.f1 =='50.50':
                p.resetJointState(self.hand_id, 0, -.725)
                p.resetJointState(self.hand_id, 1, 1.45)
            elif self.f1 == '65.35':
                p.resetJointState(self.hand_id, 0, -.46)
                p.resetJointState(self.hand_id, 1, 1.5)
            if self.f2 =='50.50':
                p.resetJointState(self.hand_id, 3, .725)
                p.resetJointState(self.hand_id, 4, -1.45)
            elif self.f2 == '65.35':
                p.resetJointState(self.hand_id, 3, .46)
                p.resetJointState(self.hand_id, 4, -1.5)
        else:
            raise KeyError('width other than 53 are not accepted at this time')
   
        p.resetBasePositionAndOrientation(self.obj_id, posObj=[0.0+obj_change[0], 0.10+obj_change[1], .05], ornObj=[0,0,0,1])

        if self.rand_start:
            f1_pos = [0.026749999999999996+obj_change[0], 0.10778391676312778+obj_change[1], 0.05]
            f2_pos = [-0.026749999999999996+obj_change[0], 0.10778391676312778+obj_change[1], 0.05]
            f1_angs = p.calculateInverseKinematics(self.hand_id, 2, f1_pos, maxNumIterations=3000)
            f2_angs = p.calculateInverseKinematics(self.hand_id, 5, f2_pos, maxNumIterations=3000)
            p.resetJointState(self.hand_id, 0, -np.pi/2)
            p.resetJointState(self.hand_id, 1, np.pi/4)
            p.resetJointState(self.hand_id, 3, np.pi/2)
            p.resetJointState(self.hand_id, 4, -np.pi/4)
            
            positions = np.linspace([-np.pi/2,np.pi/4,np.pi/2,-np.pi/4],[f1_angs[0],f1_angs[1],f2_angs[2],f2_angs[3]],20)
            for action_to_execute in positions:
                p.setJointMotorControlArray(self.hand_id, jointIndices=self.hand.get_joint_numbers(),
                                            controlMode=p.POSITION_CONTROL, targetPositions=action_to_execute,
                                            positionGains=[0.8,0.8,0.8,0.8], forces=[0.4,0.4,0.4,0.4])
                self.step()
        if self.rand_finger_pos:
            y_change = np.random.uniform(-0.01,0.01,2)
            link1_pose = p.getLinkState(self.hand_id, 2)[0]
            link2_pose = p.getLinkState(self.hand_id, 5)[0]
            f1_pos = [link1_pose[0], link1_pose[1] + y_change[0], 0.05]
            f2_pos = [link2_pose[0], link2_pose[1] + y_change[1], 0.05]
            # p.gofuckyuorself
            f1_angs = p.calculateInverseKinematics(self.hand_id, 2, f1_pos, maxNumIterations=3000)
            f2_angs = p.calculateInverseKinematics(self.hand_id, 5, f2_pos, maxNumIterations=3000)
            p.resetJointState(self.hand_id, 0, -np.pi/2)
            p.resetJointState(self.hand_id, 1, np.pi/4)
            p.resetJointState(self.hand_id, 3, np.pi/2)
            p.resetJointState(self.hand_id, 4, -np.pi/4)
            # print('wtf')

            positions = np.linspace([-np.pi/2,np.pi/4,np.pi/2,-np.pi/4],[f1_angs[0],f1_angs[1],f2_angs[2],f2_angs[3]],20)
            for action_to_execute in positions:
                p.setJointMotorControlArray(self.hand_id, jointIndices=self.hand.get_joint_numbers(),
                                            controlMode=p.POSITION_CONTROL, targetPositions=action_to_execute,
                                            positionGains=[0.8,0.8,0.8,0.8], forces=[0.4,0.4,0.4,0.4])
                self.step()
        f1_dist = p.getClosestPoints(self.obj.id, self.hand.id, 10, -1, 1, -1)
        f2_dist = p.getClosestPoints(self.obj.id, self.hand.id, 10, -1, 4, -1)
        print('f1 dist', max(f1_dist[0][8], 0))
        print('f2_dist', max(f2_dist[0][8], 0))
        

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
        