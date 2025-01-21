import pybullet as p

from mojograsp.simobjects.object_base import ActuatedObject
from mojograsp.simobjects.sensor_base import SensorBase
from pybullet_utils.bullet_client import BulletClient


# TODO: Make convenience functions for end effectors
class TwoFingerGripper(ActuatedObject):
    """TwoFingerGripper Class is a child class of ActuatedObject"""

    def __init__(self, id: int = None, path: str = None, name: str = "two_finger_gripper", physicsClientId:BulletClient=None, hand_params=None):
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
        super().__init__(id=id, path=path, name=name, physicsClientId=physicsClientId)
        self.sensor_dict = {}
        self.num_joints = p.getNumJoints(self.id)
        self.create_sensor_index()
        self.link_lengths = hand_params['link_lengths']
        self.starting_angles = hand_params['starting_angles']
        self.palm_width = hand_params['palm_width']
        self.record_name = hand_params['hand_name']

    def create_sensor_index(self):
        """
        Method is a rough draft of the create sensor index method. In the future this will construct a dictionary 
        of sensors on the object. 
        """
        print("creating sensors", self.id)
        for i in range(p.getNumJoints(self.id)):
            info = p.getJointInfo(self.id, i)
            sensor_name = info[1].decode("ascii")
            sensor_type = info[2]
            sensor_id = info[0]

            # if any joints have name with substring sensor we create a sensor object and add it to the index
            if (sensor_name.find("sensor") != -1):
                self.sensor_dict[sensor_name] = SensorBase(
                    self.id, sensor_name)
