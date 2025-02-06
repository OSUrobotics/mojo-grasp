from mojograsp.simcore.environment import Environment
from mojograsp.simobjects.two_finger_gripper import TwoFingerGripper
from mojograsp.simobjects.object_base import ObjectBase
import numpy as np
import time
import os
import pybullet_data
from scipy.spatial.transform import Rotation as R

class MultiprocessSingleShapeEnv(Environment):
    def __init__(self,pybulletInstance, hand: TwoFingerGripper, obj: ObjectBase, hand_type , args=None, finger_points=None):
        self.hand = hand
        self.obj = obj
        self.p = pybulletInstance
        mass_link = .036
        key_nums = hand_type.split('_')
        self.f1 = key_nums[1]
        self.f2 = key_nums[2]
        self.width = key_nums[-1]
        self.rand_object_start = args['object_random_start']
        self.rand_finger_position = args['finger_random_start']
        self.rand_object_orientation = args['object_random_orientation']
        self.rand_finger_all_open = args['finger_random_off']
        self.finger_open_fraction = args['fobfreq']
        try:
            # print("first try except")
            self.HIGH_FRICTION = args['friction_experiment']
            self.lateral_low = args['lat_fric_low']
            self.lateral_high = args['lat_fric_high']
            self.spinning_low = args['spin_fric_low']
            self.spinning_high = args['spin_fric_high']
            self.rolling_low = args['roll_fric_low']
            self.rolling_high = args['roll_fric_high']
            self.collision = args['collision_on']
        except:
            pass

        
        if finger_points is None:
            self.finger_points = finger_points
        else:
            print('WE SHOULD NOT BE HERE IF YOU SEE THIS SHIT WENT DOWN')
            self.finger_points = []
            self.finger_points.append(self.p.createConstraint(finger_points[0].id,-1,-1,-1,self.p.JOINT_FIXED,[0,0,0],[0,0,0],[0,0,1]))
            self.finger_points.append(self.p.createConstraint(finger_points[1].id,-1,-1,-1,self.p.JOINT_FIXED,[0,0,0],[0,0,0],[0,0,1]))
        
        self.p.resetSimulation()

        self.plane_id = self.p.loadURDF("plane.urdf", flags=self.p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES, basePosition=[0.5,0.3,0])

        #See if the paramater is there for self collision
        try:
            if self.collision:
                print('WE ARE USING SELF COLLISION')
                self.hand_id = self.p.loadURDF(self.hand.path, useFixedBase=False,
                                    basePosition=[0.0, 0.0, 0.05], flags=self.p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES | self.p.URDF_USE_SELF_COLLISION)
            else:
                print("This is strange")
                self.hand_id = self.p.loadURDF(self.hand.path, useFixedBase=False,
                                    basePosition=[0.0, 0.0, 0.05], flags=self.p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
                
        #If collision paramater is not there just use self collision
        except:
            print('NO COLLISION PARAMATER, USING DEFAULT SELF COLLISION')
            self.hand_id = self.p.loadURDF(self.hand.path, useFixedBase=False,
                                basePosition=[0.0, 0.0, 0.05], flags=self.p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES | self.p.URDF_USE_SELF_COLLISION)
        
        self.p.createConstraint(self.hand_id, -1, -1, -1, self.p.JOINT_FIXED, [0,0,0], [0,0,0], [0,0,0.05])


        self.obj_id = self.p.loadURDF(self.obj.path, basePosition=[0.0, 0.10, .05], useMaximalCoordinates=True,
                                 flags=self.p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
        # self.obj_id = self.p.loadSoftBody("/home/ubuntu/Mojograsp/mojo-grasp/demos/rl_demo/resources/object_models/Jeremiah_Shapes/Shapes/torus_textured.obj",
        #                                   simFileName="/home/ubuntu/Mojograsp/mojo-grasp/demos/rl_demo/resources/object_models/Jeremiah_Shapes/Shapes/torus.vtk",
        #                                   basePosition=[1.5, 1.5, -10.5],
        #                                   mass = 0.1,
        #                                   scale = 1,
        #                                   useNeoHookean = 1, 
        #                                   NeoHookeanMu = 180, 
        #                                   NeoHookeanLambda = 600, 
        #                                   NeoHookeanDamping = 0.01, 
        #                                   collisionMargin = 0.006, 
        #                                   useSelfCollision = 0, 
        #                                   frictionCoeff = 0.5, 
        #                                   repulsionStiffness = 800)


        # print('object path', self.obj.path)
        # assert 1==0
        try:
            print('trying to set high friction')
            if self.HIGH_FRICTION:
                self.finger_lateral_friction_range = [self.lateral_low, self.lateral_high] #[1,2] #[0.25, 0.75]
                self.finger_spinning_friction_range = [self.spinning_low, self.spinning_high] #[0.05,0.06] #[0.01,0.0101]
                self.finger_rolling_friction_range = [self.rolling_low, self.spinning_high] #[0.1,0.2] #[0.04,0.0401]
            else:
                self.finger_lateral_friction_range = [0.25, 0.75] #[1,2] #[0.25, 0.75]
                self.finger_spinning_friction_range = [0.01,0.0101] #[0.05,0.06] #[0.01,0.0101]
                self.finger_rolling_friction_range = [0.04,0.0401] #[0.1,0.2] #[0.04,0.0401]
            
        except:
            print('no high friction')
            self.finger_lateral_friction_range = [0.25, 0.75] #[1,2] #[0.25, 0.75]
            self.finger_spinning_friction_range = [0.01,0.0101] #[0.05,0.06] #[0.01,0.0101]
            self.finger_rolling_friction_range = [0.04,0.0401] #[0.1,0.2] #[0.04,0.0401]

        self.floor_lateral_friction_range = [0.15,0.25]#[0.15,0.45]
        self.floor_spinning_friction_range = [0.01,0.0101]
        self.floor_rolling_friction_range = [0.05,0.0501]
        self.object_mass_range = [0.015, 0.045]
        start_finger_lateral = np.average(self.finger_lateral_friction_range)
        start_finger_spin = np.average(self.finger_spinning_friction_range)
        start_finger_roll = np.average(self.finger_rolling_friction_range)
        start_floor_lateral = np.average(self.floor_lateral_friction_range)
        start_floor_spin = np.average(self.floor_spinning_friction_range)
        start_floor_roll = np.average(self.floor_rolling_friction_range)
        start_mass = np.average(self.object_mass_range)
        self.p.changeDynamics(self.hand_id, 1, lateralFriction=start_finger_lateral, rollingFriction=start_finger_roll, spinningFriction=start_finger_spin,
                         mass=.036)
        self.p.changeDynamics(self.hand_id, 4, lateralFriction=start_finger_lateral, rollingFriction=start_finger_roll, spinningFriction=start_finger_spin,
                         mass=.036)
        self.p.changeDynamics(self.hand_id, 0, jointLowerLimit=-1.57, jointUpperLimit=1.57, mass=mass_link)
        self.p.changeDynamics(self.hand_id, 1, jointLowerLimit=0, jointUpperLimit=2.09, mass=mass_link)
        self.p.changeDynamics(self.hand_id, 3, jointLowerLimit=-1.57, jointUpperLimit=1.57, mass=mass_link)
        self.p.changeDynamics(self.hand_id, 4, jointLowerLimit=-2.09, jointUpperLimit=0, mass=mass_link)
        self.p.changeDynamics(self.plane_id,-1,lateralFriction=start_floor_lateral, spinningFriction=start_floor_spin, rollingFriction=start_floor_roll)
        self.p.changeDynamics(self.obj.id, -1, mass=start_mass, restitution=.95, lateralFriction=1, localInertiaDiagonal=[0.000029435425,0.000029435425,0.00000725805])
        self.p.changeVisualShape(self.hand_id, -1, rgbaColor=[0.3, 0.3, 0.3, 1])
        self.p.changeVisualShape(self.hand_id, 0, rgbaColor=[1, 0.5, 0, 1])
        self.p.changeVisualShape(self.hand_id, 1, rgbaColor=[0.3, 0.3, 0.3, 1])
        self.p.changeVisualShape(self.hand_id, 3, rgbaColor=[1, 0.5, 0, 1])
        self.p.changeVisualShape(self.hand_id, 4, rgbaColor=[0.3, 0.3, 0.3, 1])
        self.p.changeVisualShape(self.obj_id, -1, rgbaColor=[0.1, 0.6, 0.1, 1])
        # self.p.configureDebugVisualizer(self.p.COV_ENABLE_SHADOWS, 0)
        self.p.configureDebugVisualizer(self.p.COV_ENABLE_GUI, 0)
        self.hand.id = self.hand_id
        self.obj.id = self.obj_id
        self.start_time = 0
        # self.p.setPhysicsEngineParameter(enableFileCaching=1) 
        self.p.setGravity(0, 0, -10)
        self.p.setPhysicsEngineParameter(contactBreakingThreshold=.001, contactERP = 0.35, globalCFM = 0.0000001 ,numSubSteps=2)#, useSplitImpulse=1)
        self.p.setRealTimeSimulation(0)
        fixed=False
        if fixed:
            self.p.createConstraint(self.obj_id, -1, -1, -1, self.p.JOINT_POINT2POINT, [0, 0, 1],
                       [0, 0, 0], [0, 0.1, 0])
            
        # need to update friction values
        # print('startng object mass', start_mass)
        # print('starting floor frictions', start_floor_lateral, start_floor_spin, start_floor_roll)
        # print('starting finger frictions',start_finger_lateral, start_finger_spin, start_finger_roll)

    def set_friction(self, frictionList):
        #print("Friction list in env is :", frictionList)
        self.p.changeDynamics(self.hand_id, -1,lateralFriction=frictionList[0], spinningFriction=frictionList[1], rollingFriction=frictionList[2])
        self.p.changeDynamics(self.plane_id, -1,lateralFriction=frictionList[3], spinningFriction=frictionList[4], rollingFriction=frictionList[5])
        self.p.changeDynamics(self.obj.id, -1, mass=.03, restitution=.95, lateralFriction=frictionList[6],spinningFriction=frictionList[7], rollingFriction=frictionList[8])


    def set_contact(self,contactList):
        self.p.changeDynamics(self.hand_id, 1, contactStiffness=contactList[0], contactDamping=contactList[1])#, restitution=contactList[2])
        self.p.changeDynamics(self.hand_id, 4, contactStiffness=contactList[0], contactDamping=contactList[1])#, restitution=contactList[2])


    def make_viz_point(self,thing):
        if type(thing[0]) == list:
            for i in thing:
                temp = self.p.loadURDF("sphere_1cm.urdf", basePosition=i, baseOrientation=[0, 0, 0, 1], globalScaling=0.25)
                self.p.changeVisualShape(temp,-1,rgbaColor=[0,0,1,1])
        else:
            temp=self.p.loadURDF("sphere_1cm.urdf", basePosition=thing, baseOrientation=[0, 0, 0, 1], globalScaling=0.5)
            self.p.changeVisualShape(temp,-1,rgbaColor=[1,0,0,1])


    def reset(self, obj_dict=None,finger_dict=None):
        # reset the simulator
        # self.p.resetSimulation()

        # self.p.setPhysicsEngineParameter(enableFileCaching=1)

        # Goal of this redesign is to shift all the randomization to outside of this function
        # this should accept object pos and finger pos and run from there
        # takes two dictionaries. object_pos {'translation':[x,y], 'rotation':theta}
        # finger_pos {'joint_angles':[t1,t2,t3,t4]} OR {'finger_y':[f1y,f2y]}
        if obj_dict is not None:
            obj_xy = obj_dict['translation']
            obj_theta = obj_dict['rotation']
        else:
            obj_xy = np.array([0,0])
            obj_theta = 0

        if finger_dict is None:
            angles = []
        elif 'joint_angles' in finger_dict.keys():
            angles = finger_dict['joint_angles']
        elif 'finger_y' in finger_dict.keys():
            link1_pose = self.p.getLinkState(self.hand_id, 2)[0]
            link2_pose = self.p.getLinkState(self.hand_id, 5)[0]
            # print('all relevant things', link1_pose, obj_change,y_change)
            f1_pos = [link1_pose[0]+obj_xy[0], link1_pose[1] + obj_xy[1] + finger_dict['finger_y'][0], 0.05]
            f2_pos = [link2_pose[0]+obj_xy[0], link2_pose[1] + obj_xy[1] + finger_dict['finger_y'][1], 0.05]
            f1_angs = self.p.calculateInverseKinematics(self.hand_id, 2, f1_pos, maxNumIterations=3000)
            f2_angs = self.p.calculateInverseKinematics(self.hand_id, 5, f2_pos, maxNumIterations=3000)
            angles = [f1_angs[0],f1_angs[1],f2_angs[2],f2_angs[3]]
        obj_quat = R.from_euler('xyz',[0,0,obj_theta]).as_quat()
        self.p.resetJointState(self.hand.id, 0, self.hand.starting_angles[0])
        self.p.resetJointState(self.hand.id, 1, self.hand.starting_angles[1])
        self.p.resetJointState(self.hand.id, 3, self.hand.starting_angles[2])
        self.p.resetJointState(self.hand.id, 4, self.hand.starting_angles[3])
        
        # print('starting angs', self.hand.starting_angles)
        self.p.resetBasePositionAndOrientation(self.obj_id, posObj=[0.0+obj_xy[0], 0.10+obj_xy[1], .05], ornObj=obj_quat)
        self.p.resetBaseVelocity(self.obj_id, [0,0,0], [0,0,0])

        # print('in the first')
        self.p.resetJointState(self.hand_id, 0, -np.pi/2)
        self.p.resetJointState(self.hand_id, 1, np.pi/4)
        self.p.resetJointState(self.hand_id, 3, np.pi/2)
        self.p.resetJointState(self.hand_id, 4, -np.pi/4)
        positions = np.linspace([-np.pi/2,np.pi/4,np.pi/2,-np.pi/4],[angles[0],angles[1],angles[2],angles[3]],20)
        real_positions = np.ones((30,4))
        real_positions[0:20,:] = positions
        real_positions[20:,:] = angles
        for action_to_execute in real_positions:
            self.p.setJointMotorControlArray(self.hand_id, jointIndices=self.hand.get_joint_numbers(),
                                        controlMode=self.p.POSITION_CONTROL, targetPositions=action_to_execute,
                                        positionGains=[0.8,0.8,0.8,0.8], forces=[0.4,0.4,0.4,0.4])
            self.step()

    def apply_domain_randomization(self, finger_friction, floor_friction, object_mass):
        # print('dr terms',finger_friction, floor_friction, object_mass)
        if object_mass:
            new_mass = np.random.uniform(self.object_mass_range[0], self.object_mass_range[1])
            # print('object mass', new_mass)
            self.p.changeDynamics(self.obj_id, -1, mass=new_mass)
        if floor_friction:
            new_lateral_friction = np.random.uniform(self.floor_lateral_friction_range[0],self.floor_lateral_friction_range[1])
            new_spinning_friction = np.random.uniform(self.floor_spinning_friction_range[0],self.floor_spinning_friction_range[1])
            new_rolling_friction = np.random.uniform(self.floor_rolling_friction_range[0],self.floor_rolling_friction_range[1])
            # print('floor frictions', new_lateral_friction, new_spinning_friction, new_rolling_friction)
            self.p.changeDynamics(self.plane_id, -1, lateralFriction=new_lateral_friction, spinningFriction=new_spinning_friction, rollingFriction=new_rolling_friction)
        if finger_friction:
            new_lateral_friction = np.random.uniform(self.finger_lateral_friction_range[0],self.finger_lateral_friction_range[1])
            new_spinning_friction = np.random.uniform(self.finger_spinning_friction_range[0],self.finger_spinning_friction_range[1])
            new_rolling_friction = np.random.uniform(self.finger_rolling_friction_range[0],self.finger_rolling_friction_range[1])
            # print('finger frictions', new_lateral_friction, new_spinning_friction, new_rolling_friction)
            self.p.changeDynamics(self.hand_id, 1, lateralFriction=new_lateral_friction, spinningFriction=new_spinning_friction)
            self.p.changeDynamics(self.hand_id, 4, lateralFriction=new_lateral_friction, spinningFriction=new_spinning_friction, rollingFriction=new_rolling_friction)

    def reset_to_pos(self, object_pos, finger_angles):
        # reset the simulator
        self.p.resetSimulation()
        # reload the objects
        plane_id = self.p.loadURDF("plane.urdf", flags=self.p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)

        # For alt configuration
        hand_id = self.p.loadURDF(self.hand.path, useFixedBase=True,
                             basePosition=[0.0, 0.0, 0.05], flags=self.p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES | self.p.URDF_USE_SELF_COLLISION)

        self.p.resetJointState(hand_id, 0, finger_angles[0])
        self.p.resetJointState(hand_id, 1, finger_angles[1])
        self.p.resetJointState(hand_id, 3, finger_angles[2])
        self.p.resetJointState(hand_id, 4, finger_angles[3])

        mass_link = .036
        #ASK ABOUT THIS
        self.p.changeDynamics(hand_id, 1, lateralFriction=self.lateral_low , rollingFriction=self.rolling_low,
                         mass=.036)
        self.p.changeDynamics(hand_id, 4, lateralFriction=self.lateral_low, rollingFriction=self.rolling_low,
                         mass=.036)
        self.p.changeDynamics(hand_id, 0, jointLowerLimit=-1.57, jointUpperLimit=1.57, mass=mass_link)
        self.p.changeDynamics(hand_id, 1, jointLowerLimit=0, jointUpperLimit=2.09, mass=mass_link, contactStiffness=10, contactDamping=0.1)
        self.p.changeDynamics(hand_id, 3, jointLowerLimit=-1.57, jointUpperLimit=1.57, mass=mass_link)
        self.p.changeDynamics(hand_id, 4, jointLowerLimit=-2.09, jointUpperLimit=0, mass=mass_link, contactStiffness=10, contactDamping=0.1)
        
        noisex = np.random.normal(0, 0.0015, 2)
        noisey = np.random.normal(0, 0.0015, 2)

        obj_id = self.p.loadURDF(self.obj.path, basePosition=[object_pos[0] + noisex, object_pos[1] + noisey, .05], useMaximalCoordinates=True,
                        flags=self.p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
        # obj_id = self.p.loadSoftBody("/home/ubuntu/Mojograsp/mojo-grasp/demos/rl_demo/resources/object_models/Jeremiah_Shapes/Shapes/torus.obj",
        #                                   scale = 0.1,
        #                                   basePosition=[1.5, 1.5, -10.5],
        #                                   mass = 0.1)


        self.p.changeDynamics(plane_id,-1,lateralFriction=0.5, spinningFriction=0.01, rollingFriction=0.05)
        self.p.changeDynamics(self.obj.id, -1, mass=.03, restitution=.95, lateralFriction=0.5)
        
        self.p.setGravity(0, 0, -10)
        self.p.setPhysicsEngineParameter(contactBreakingThreshold=.001, contactERP = 0.35, globalCFM = 0.0000001 ,numSubSteps=2)
        self.p.p.setRealTimeSimulation(0)


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
        # temp =self.p.getContactPoints(self.hand_id, self.obj_id)
        # if temp != ():
        #     if temp[0][8] < -0.8/1000:
        #         print('contact points', temp[0][8] * 1000)
            
        
    def set_finger_contact_goal(self,finger_goals):
        if self.finger_points is None:
            raise EnvironmentError('Tried to set finger goal points in an environment without finger goal objects')
        else:
            for finger,goal in zip(self.finger_points,finger_goals):
                self.p.changeConstraint(finger,goal)

    def set_obj_pose(self,pose):
        current_obj_pose = self.p.getBasePositionAndOrientation(self.obj_id)
        self.p.resetBasePositionAndOrientation(self.obj_id, posObj=[pose[0][0],pose[0][1],current_obj_pose[0][2]], ornObj=pose[1])



class MultiprocessMazeEnv(MultiprocessSingleShapeEnv):
    def __init__(self, pybulletInstance, hand: TwoFingerGripper, obj: ObjectBase, wall: ObjectBase, goal_block, hand_type, args=None, finger_points=None):
        super().__init__(pybulletInstance, hand, obj, hand_type, args, finger_points)
        # print(self.p.getBaseVelocity(self.obj_id))
        # print('checking something')
        self.wall = wall
        self.goals = goal_block
        self.wall_id = self.p.loadURDF(self.wall.path, basePosition=[0,0.02,0.02],
                                 flags=self.p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)

        self.p.setCollisionFilterPair(self.wall_id, self.hand_id,-1,0,0)
        self.p.setCollisionFilterPair(self.wall_id, self.hand_id,-1,1,0)
        self.p.setCollisionFilterPair(self.wall_id, self.hand_id,-1,2,0)
        self.p.setCollisionFilterPair(self.wall_id, self.hand_id,-1,3,0)
        self.p.setCollisionFilterPair(self.wall_id, self.hand_id,-1,4,0)
        self.p.setCollisionFilterPair(self.wall_id, self.hand_id,-1,5,0)
        self.p.setCollisionFilterPair(self.wall_id,self.obj_id, -1, -1, 0)
        self.p.changeVisualShape(self.wall_id, -1, rgbaColor=[0.2, 0.2, 1, 1])
        # print('total constraints',self.p.getNumConstraints())
        self.wall.init_constraint([0,0.02,0.02],[0,0,0,1])

    def set_goal(self,goal):
        self.goals.set_goal(goal)
        self.wall.set_curr_pose([0,0.01,0.02],[0,0,0,1])
        # self.p.setCollisionFilterPair(self.wall_id, self.obj_id,-1,-1,0)

    def set_wall_pose(self,pose):
        self.wall.set_pose(pose)
