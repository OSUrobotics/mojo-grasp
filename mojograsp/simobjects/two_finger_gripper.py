import pybullet as p

from mojograsp.simobjects.object_base import ActuatedObject
from mojograsp.simobjects.sensor_base import SensorBase


# TODO: Make convenience functions for end effectors
class TwoFingerGripper(ActuatedObject):

    def __init__(self, id: int = None):
        super().__init__(id=id)
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

    def get_joint_angles(self, joint_numbers: list[int] = None) -> list[float]:
        """
        Get the current pose angle of joints
        Stores in self.curr_joint_angle : current joint angles as a list
        :param joint_indices: List of particular joint indices to get angles for. If None, returns all joint angle values.
        """
        curr_joint_poses = []
        if joint_numbers is None:
            curr_joint_states = p.getJointStates(
                self.id, self.joint_dict.values())
        else:
            curr_joint_states = p.getJointStates(self.id, joint_numbers)
        for joint in range(0, len(curr_joint_states)):
            curr_joint_poses.append(curr_joint_states[joint][0])
        return curr_joint_poses
