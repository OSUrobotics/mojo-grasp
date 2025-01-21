from mojograsp.simcore.environment import Environment
from mojograsp.simobjects.two_finger_gripper import TwoFingerGripper
from mojograsp.simobjects.object_base import ObjectBase
import numpy as np
import time
import os

class MultiprocessDirectionSingleShapeEnv(Environment):
    def __init__(self, pybulletInstance, hand: TwoFingerGripper, obj: ObjectBase, args=None, obj_starts=[[0,0]], finger_ys=[[0,0]]):
        self.hand = hand
        self.obj = obj
        self.p = pybulletInstance
        mass_link = .036
        self.rand_object_start = args['object_random_start']
        self.rand_finger_position = args['finger_random_start']
        self.rand_object_orientation = args['object_random_orientation']
        self.rand_finger_all_open = args['finger_random_off']
        self.finger_open_fraction = args['fobfreq']

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
        self.p.changeDynamics(self.plane_id,-1,lateralFriction=0.5, spinningFriction=0.01, rollingFriction=0.05)
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
        
        self.count = 0
        self.object_start_poses = obj_starts
        self.finger_ys = finger_ys
        self.obj_len = len(obj_starts)
        self.fing_len = len(finger_ys)
        self.shuffle_len = max(self.obj_len,self.fing_len)
        self.p.setGravity(0, 0, -10)
        self.p.setPhysicsEngineParameter(contactBreakingThreshold=.001)

    def reset(self):
        # reset the simulator        
        # this env is designed to hold its own object start pose and finger y start pose
        obj_change = self.object_start_poses[self.count%self.obj_len]
        y_change = self.finger_ys[self.count%self.fing_len]
        self.p.resetJointState(self.hand.id, 0, self.hand.starting_angles[0])
        self.p.resetJointState(self.hand.id, 1, self.hand.starting_angles[1])
        self.p.resetJointState(self.hand.id, 3, self.hand.starting_angles[2])
        self.p.resetJointState(self.hand.id, 4, self.hand.starting_angles[3])
        
        self.p.resetBasePositionAndOrientation(self.obj_id, posObj=[0.0+obj_change[0], 0.10+obj_change[1], .05], ornObj=[0,0,0,1])
        link1_pose = self.p.getLinkState(self.hand_id, 2)[0]
        link2_pose = self.p.getLinkState(self.hand_id, 5)[0]
        f1_pos = [link1_pose[0]+obj_change[0], link1_pose[1] + obj_change[1] + y_change[0], 0.05]
        f2_pos = [link2_pose[0]+obj_change[0], link2_pose[1] + obj_change[1] + y_change[1], 0.05]
        f1_angs = self.p.calculateInverseKinematics(self.hand_id, 2, f1_pos, maxNumIterations=3000)
        f2_angs = self.p.calculateInverseKinematics(self.hand_id, 5, f2_pos, maxNumIterations=3000)
        self.p.resetJointState(self.hand_id, 0, -np.pi/2)
        self.p.resetJointState(self.hand_id, 1, np.pi/4)
        self.p.resetJointState(self.hand_id, 3, np.pi/2)
        self.p.resetJointState(self.hand_id, 4, -np.pi/4)
        
        positions = np.linspace([-np.pi/2,np.pi/4,np.pi/2,-np.pi/4],[f1_angs[0],f1_angs[1],f2_angs[2],f2_angs[3]],20)
        real_positions = np.ones((30,4))
        real_positions[0:20,:] = positions
        real_positions[20:,:]=[f1_angs[0],f1_angs[1],f2_angs[2],f2_angs[3]]
        for action_to_execute in real_positions:
            self.p.setJointMotorControlArray(self.hand_id, jointIndices=self.hand.get_joint_numbers(),
                                        controlMode=self.p.POSITION_CONTROL, targetPositions=action_to_execute,
                                        positionGains=[0.8,0.8,0.8,0.8], forces=[0.4,0.4,0.4,0.4])
            self.step()

        # thing = self.p.getBaseVelocity(self.obj_id)
        self.count+= 1
        if self.count > self.shuffle_len:
            self.count = 0
            np.random.shuffle(self.object_start_poses)
            np.random.shuffle(self.finger_ys)
            

    def setup(self):
        super().setup()

    def step(self):
        super().step()
        
    def set_finger_contact_goal(self,finger_goals):
        if self.finger_points is None:
            raise EnvironmentError('Tried to set finger goal points in an environment without finger goal objects')
        else:
            for finger,goal in zip(self.finger_points,finger_goals):
                self.p.changeConstraint(finger,goal)

class MultiprocessDirectionUpperEnv(Environment):
    def __init__(self, pybulletInstance, hand: TwoFingerGripper, obj: ObjectBase, args=None, obj_starts=[[0,0]], finger_ys=[[0,0]]):
        self.hand = hand
        self.obj = obj
        self.p = pybulletInstance
        mass_link = .036
        self.rand_object_start = args['object_random_start']
        self.rand_finger_position = args['finger_random_start']
        self.rand_object_orientation = args['object_random_orientation']
        self.rand_finger_all_open = args['finger_random_off']
        self.finger_open_fraction = args['fobfreq']

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
        self.p.changeDynamics(self.plane_id,-1,lateralFriction=0.5, spinningFriction=0.01, rollingFriction=0.05)
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
        
        self.count = 0
        self.object_start_poses = obj_starts
        self.finger_ys = finger_ys
        self.obj_len = len(obj_starts)
        self.fing_len = len(finger_ys)
        self.shuffle_len = max(self.obj_len,self.fing_len)
        self.p.setGravity(0, 0, -10)
        self.p.setPhysicsEngineParameter(contactBreakingThreshold=.001)

    def reset(self):
        # reset the simulator        
        # this env is designed to hold its own object start pose and finger y start pose
        obj_change = self.object_start_poses[self.count%self.obj_len]
        y_change = self.finger_ys[self.count%self.fing_len]
        self.p.resetJointState(self.hand.id, 0, self.hand.starting_angles[0])
        self.p.resetJointState(self.hand.id, 1, self.hand.starting_angles[1])
        self.p.resetJointState(self.hand.id, 3, self.hand.starting_angles[2])
        self.p.resetJointState(self.hand.id, 4, self.hand.starting_angles[3])
        
        self.p.resetBasePositionAndOrientation(self.obj_id, posObj=[0.0+obj_change[0], 0.10+obj_change[1], .05], ornObj=[0,0,0,1])
        link1_pose = self.p.getLinkState(self.hand_id, 2)[0]
        link2_pose = self.p.getLinkState(self.hand_id, 5)[0]
        f1_pos = [link1_pose[0]+obj_change[0], link1_pose[1] + obj_change[1] + y_change[0], 0.05]
        f2_pos = [link2_pose[0]+obj_change[0], link2_pose[1] + obj_change[1] + y_change[1], 0.05]
        f1_angs = self.p.calculateInverseKinematics(self.hand_id, 2, f1_pos, maxNumIterations=3000)
        f2_angs = self.p.calculateInverseKinematics(self.hand_id, 5, f2_pos, maxNumIterations=3000)
        self.p.resetJointState(self.hand_id, 0, -np.pi/2)
        self.p.resetJointState(self.hand_id, 1, np.pi/4)
        self.p.resetJointState(self.hand_id, 3, np.pi/2)
        self.p.resetJointState(self.hand_id, 4, -np.pi/4)
        
        positions = np.linspace([-np.pi/2,np.pi/4,np.pi/2,-np.pi/4],[f1_angs[0],f1_angs[1],f2_angs[2],f2_angs[3]],20)
        real_positions = np.ones((30,4))
        real_positions[0:20,:] = positions
        real_positions[20:,:]=[f1_angs[0],f1_angs[1],f2_angs[2],f2_angs[3]]
        for action_to_execute in real_positions:
            self.p.setJointMotorControlArray(self.hand_id, jointIndices=self.hand.get_joint_numbers(),
                                        controlMode=self.p.POSITION_CONTROL, targetPositions=action_to_execute,
                                        positionGains=[0.8,0.8,0.8,0.8], forces=[0.4,0.4,0.4,0.4])
            self.step()

        # thing = self.p.getBaseVelocity(self.obj_id)
        self.count+= 1
        if self.count > self.shuffle_len:
            self.count = 0
            np.random.shuffle(self.object_start_poses)
            np.random.shuffle(self.finger_ys)
            

    def setup(self):
        super().setup()

    def step(self):
        super().step()
        
    def set_finger_contact_goal(self,finger_goals):
        if self.finger_points is None:
            raise EnvironmentError('Tried to set finger goal points in an environment without finger goal objects')
        else:
            for finger,goal in zip(self.finger_points,finger_goals):
                self.p.changeConstraint(finger,goal)