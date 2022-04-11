import pybullet as p

from mojograsp.simobjects.object_base import ActuatedObject
from mojograsp.simobjects.sensor_base import SensorBase


# TODO: Make convenience functions for end effectors
class TwoFingerGripper(ActuatedObject):

    def __init__(self, id: int = None, path: str = None, name: str = "two_finger_gripper"):
        super().__init__(id=id, path=path, name=name)
        self.sensor_dict = {}
        self.num_joints = p.getNumJoints(self.id)
        self.create_sensor_index()

    # rough draft of sensor index creation
    def create_sensor_index(self):
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
