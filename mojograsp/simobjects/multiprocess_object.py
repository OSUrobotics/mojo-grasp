from numpy import angle
import logging
from pybullet_utils.bullet_client import BulletClient
from scipy.spatial.transform import Rotation as R

class MultiprocessObjectBase:
    """ObjectBase Base Class"""

    def __init__(self, physicsClientId, id: int = None, path: str = None, name: str = None):
        """
        Constructor takes in object id, path to urdf or sdf files (If one exists) and an object name. 
        This serves as the base class that all other object types should inherit from. 

        :param id: The id returned from pybullet when an object is loaded in. 
        :param path: Path to the urdf or sdf file that was used. 
        :param name: Name of the object.  
        :type id: int
        :type path: str
        :type name: str
        """
        # all coordinate frame magic will be done here.
        logging.info("Object Created with id: {}".format(id))
        self.path = path
        self.id = id
        self.p = physicsClientId
        self.name = name
        if not self.name:
            self.name = "obj_" + str(id)

    def get_curr_pose(self) -> list:
        """
        Gets the current base position and orientation, returns a list of 2 lists: [[position],[orientation]]

        :return: list of lists [[position],[orientation]]
        :rtype: list[list]
        """
        pose = []
        curr_pos, curr_orn = self.p.getBasePositionAndOrientation(self.id)
        pose.append(list(curr_pos))
        pose.append(list(curr_orn))
        return pose

    def get_dimensions(self) -> list:
        """
        Returns the visual shape data dimensions of an object.

        :return: list
        :rtype: list
        """
        return self.p.getVisualShapeData(self.id)[0][3]

    def set_curr_pose(self, pos, orn):
        """
        Sets the current position and orientation of an object. Should not be used while stepping the simulator, 
        only before or after a trial. 

        :param pos: position [x,y,z]
        :param orn: orientation (quaternion) [x,y,z,w]
        :type pos: list
        :type orn: list
        """
        if len(orn) == 3:
            orn = self.p.getQuaternionFromEuler(orn)
        self.p.resetBasePositionAndOrientation(self.id, pos, orn)

    def get_curr_velocity(self) -> list:
        """
        Gets the current linear velocity and angular velocity, returns a list of lists: [[linear velocity],[angular velocity]]

        :return: list [[linear velocity],[angular velocity]]
        :rtype: list[list]
        """
        vel = []
        linear_velocity, angular_velocity = self.p.getBaseVelocity(self.id)
        vel.append(list(linear_velocity))
        vel.append(list(angular_velocity))
        return vel

# TODO: This should be more fleshed out in the future to get all relevant data
    def get_data(self) -> dict:
        """
        Placeholder method that should be overriden if you would like to return more data. 
        It is used in :func:`~mojograsp.simcore.state.StateDefault` to collect the state information 
        of an object. The default dictionary that is returned contains the current pose of the object.

        :return: dictionary of data about the object (can be used with the default state class)
        :rtype: dict
        """
        data = {}
        data["pose"] = self.get_curr_pose()
        data['z_angle'] = R.from_quat(data['pose'][1]).as_euler('xyz')[-1]
        return data

# TODO: Create functions for keeping track of links similar to joints


class MultiprocessActuatedObject(MultiprocessObjectBase):
    """
    ActuatedObject Class inherits from ObjectBase and is intended to make it easier to work with 
    actuated objects with pybullet. Offers helper functions and keeps track of joint dictionary. 
    """

    def __init__(self, physicsClientId, id: int = None, path: str = None, name: str = None):
        """
        Constructor takes in object id, path to urdf or sdf files (If one exists) and an object name. 
        This serves as the base class that all other object types should inherit from. 

        :param id: The id returned from pybullet when an object is loaded in. 
        :param path: Path to the urdf or sdf file that was used. 
        :param name: Name of the object.  
        :type id: int
        :type path: str
        :type name: str
        """

        super().__init__(physicsClientId, id=id, path=path, name=name)
        self.joint_dict = {}
        self.create_joint_dict()

    def create_joint_dict(self):
        """
        Method creates the joint dictionary for the given object, containing both name and
        joint number for easier use.  
        TODO: Add link functionality.
        """
        # creates joint dictionary with style = name: number
        for i in range(self.p.getNumJoints(self.id)):
            info = self.p.getJointInfo(self.id, i)
            joint_name = info[1].decode("ascii")
            joint_type = info[2]
            joint_id = info[0]
            # filters out fixed joints
            if (joint_type != 4):
                self.joint_dict[joint_name] = joint_id
        logging.info("Joint dict: ", self.joint_dict)

    def get_joint_dict(self) -> dict:
        """
        Method returns the joint dictionary for the given object, containing both name and
        joint number for easier use.  

        :return: Joint dictionary for object
        :rtype: dict
        """
        return self.joint_dict

    def get_joint_names(self) -> list:
        """
        Method returns the joint dictionary for the given object, containing both name and
        joint number for easier use.  

        :return: Joint dictionary for object
        :rtype: dict
        """
        return list(self.joint_dict.keys())

    def get_joint_numbers(self) -> list:
        """
        Method returns a list of all the joint numbers.

        :return: List of joint numbers
        :rtype: list
        """
        return list(self.joint_dict.values())

    # takes a list of names and returns the joint numbers
    def get_joints_by_name(self, names: list = []) -> list:
        """
        Method returns a list of joint numbers that correspond to the list of given joint names. 

        :param names: list of joint names that you would like the number of.
        :return: list of joint numbers.
        :rtype: list
        """
        joint_num_list = []
        for i in names:
            if(i in self.joint_dict):
                joint_num_list.append(self.joint_dict[i])
            else:
                logging.warn("Could not find " + str(i) + " in joint dict")
        return joint_num_list

    # gets joint names from a list of joint numbers
    def get_joints_by_number(self, numbers: list = []) -> list:
        """
        Method returns a list of joint names that correspond to the list of given joint numbers. 
        TODO: REDO THIS FUNCTION, IMPLEMENTED WEIRD

        :param names: list of joint numbers that you would like the name of.
        :return: list of joint names.
        :rtype: list
        """
        joint_name_list = []
        name = list(self.joint_dict.keys())
        num = list(self.joint_dict.values())
        for i in numbers:
            found = False
            for j in range(len(num)):
                if(i == num[j]):
                    joint_name_list.append(name[j])
                    found = True
                    break
            if(found == False):
                logging.warn("Could not find joint " +
                             str(i) + " in joint dict")
        return joint_name_list

    def get_joint_angles(self, joint_numbers: list = None) -> list:
        """
        Method returns a list of joint angles that correspond to the list of given joint numbers. 
        If joint_numbers is None then all joint angles are returned.

        :param joint_numbers: list of joint numbers that you would like to get the current joint angles for.
        :return: List of current joint angles.
        :rtype: list
        """
        curr_joint_poses = []
        if joint_numbers is None:
            curr_joint_states = self.p.getJointStates(
                self.id, self.joint_dict.values())
        else:
            curr_joint_states = self.p.getJointStates(self.id, joint_numbers)
        for joint in range(0, len(curr_joint_states)):
            curr_joint_poses.append(curr_joint_states[joint][0])
        return curr_joint_poses

    def get_data(self) -> dict:
        """
        Method overrides the ObjectBase Parent class get_data() to instead return a dictionary containing
        the current pose, joint names and joint angles.

        :return: dictionary containing object base pose (list), and joint_angles/names (dict)
        :rtype: dict
        """
        data = {}
        data["pose"] = self.get_curr_pose()
        names = self.get_joint_names()
        angles = self.get_joint_angles()
        angle_dict = {}
        for i in range(len(names)):
            angle_dict[names[i]] = angles[i]
        data["joint_angles"] = angle_dict
        return data

class ObjectWithVelocity(MultiprocessObjectBase):
    def __init__(self, physicsClientId, id: int = None, path: str = None, name: str = None):
        super().__init__(physicsClientId, id=id, path=path, name=name)

    def get_data(self):
        """
        It is used in :func:`~mojograsp.simcore.state.StateDefault` to collect the state information 
        of an object. The default dictionary that is returned contains the current pose of the object.

        :return: dictionary of data about the object (can be used with the default state class)
        :rtype: dict
        """
        data = {}
        data["pose"] = self.get_curr_pose()
        data["velocity"] = self.get_curr_velocity()
        return data

class MultiprocessFixedObject(MultiprocessObjectBase):
    def __init__(self, physicsClientId, id: int = None, path: str = None, name: str = None):
        super().__init__(physicsClientId, id, path, name, )
        pose = self.p.getBasePositionAndOrientation(id)
        self.constraint = self.p.createConstraint(self.id, -1, -1, -1, self.p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], pose[0], childFrameOrientation=pose[1])
    
    def init_constraint(self,pos,orn):
        self.constraint = self.p.createConstraint(self.id, -1, -1, -1, self.p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], pos, childFrameOrientation=orn)

    def set_curr_pose(self, pos, orn):
        self.p.changeConstraint(self.constraint,pos,orn)
        return super().set_curr_pose(pos, orn)