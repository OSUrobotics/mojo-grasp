import pybullet as p
from . import objectbase
from . import handgeometry

from mojograsp.simcore.actuators import fullyactuated, underactuated
from mojograsp.simcore.sensors import sensorbase


class Hand(objectbase.ObjectBase):

    def __init__(self, filename, underactuation=False, fixed=False, base_pos=None, base_orn=None):
        # calls initialization for objectbase class
        super().__init__(filename, fixed, base_pos, base_orn)

        # initialize important variables
        self.joint_index = {}
        self.sensor_index = {}
        self.num_joints = p.getNumJoints(self.id)
        self.create_sensor_index()
        self.create_joint_index()
        self.create_joint_dict()

        # creates instances of geometry class and actuators class
        self.geometry = handgeometry.HandGeometry(self.joint_index)
        if (underactuation == False):
            self.actuation = fullyactuated.FullyActuated(self.joint_index, self.id)
        else:
            self.actuation = underactuated.UnderActuated(self.joint_index)

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
                self.sensor_index[sensor_name] = sensorbase.SensorBase(self.id, sensor_name)

    # creates joint dictionary with style name: number
    def create_joint_index(self):
        print("creating joint index")
        for i in range(p.getNumJoints(self.id)):
            info = p.getJointInfo(self.id, i)
            joint_name = info[1].decode("ascii")
            joint_type = info[2]
            joint_id = info[0]
            # filters out fixed joints to exclude sensors but will need better solution in the future
            if (joint_type != 4):
                self.joint_index[joint_name] = joint_id

        print(self.joint_index)

    def create_joint_dict(self, keys=None):
        """
        Create a dictionary for easy referencing of joints
        :param keys: If None, uses nomenclature from urdf file, else uses nomenclature passed
        Note: 0 index is reserved for Base of the manipulator. Expected key for this is always 'Base'
        :return:
        """
        self.joint_dict_with_base = {}
        self.joint_dict = {}
        self.key_names_list_with_base = []
        self.key_names_list = []
        self.end_effector_indices = []
        self.prox_indices = []

        self.get_joints_info()

        for i in range(0, len(self.joint_info)):
            if i == 0:
                self.joint_dict_with_base.update({self.joint_info[i][1]: self.joint_info[i][0]})
                self.key_names_list_with_base.append(self.joint_info[i][1])
                continue
            self.joint_dict.update({self.joint_info[i][1]: self.joint_info[i][0]})
            self.joint_dict_with_base.update({self.joint_info[i][1]: self.joint_info[i][0]})
            self.key_names_list.append(self.joint_info[i][1])
            self.key_names_list_with_base.append(self.joint_info[i][1])
            if 'dist' in str(self.joint_info[i][1]):
                self.end_effector_indices.append(i)

            if 'prox' in str(self.joint_info[i][1]):
                self.prox_indices.append(i)

        print(self.prox_indices)

        # print("DICTIONARY CREATED:{}".format(self.joint_dict))
        # print("DICTIONARY CREATED WITH BASE:{}".format(self.joint_dict_with_base))
        # print("LIST CREATED:{}".format(self.key_names_list))
        print("LIST CREATED WITH BASE:{}".format(self.key_names_list_with_base))

    # print("END EFFECTOR LIST CREATED:{}".format(self.end_effector_indices))

    def get_joints_info(self):
        """
        TODO: make this private?
        Get joint info of every joint of the manipulator
        :return: list of joint info of every joint
        """
        self.joint_info = []
        for joint in range(self.num_joints):
            self.joint_info.append(p.getJointInfo(self.id, joint))
        return self.joint_info
