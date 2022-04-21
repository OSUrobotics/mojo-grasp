from numpy import angle
import pybullet as p
import logging


class ObjectBase:

    def __init__(self, id: int = None, path: str = None, name: str = None):
        # all coordinate frame magic will be done here.
        logging.info("Object Created with id: {}".format(id))
        self.path = path
        self.id = id
        self.name = name
        if not self.name:
            self.name = "obj_" + str(id)

    def get_curr_pose(self) -> list:
        pose = []
        curr_pos, curr_orn = p.getBasePositionAndOrientation(self.id)
        pose.append(list(curr_pos))
        pose.append(list(curr_orn))
        return pose

    def get_dimensions(self) -> list:
        return p.getVisualShapeData(self.id)[0][3]

    def set_curr_pose(self, pos, orn):
        if len(orn) == 3:
            orn = p.getQuaternionFromEuler(orn)
        p.resetBasePositionAndOrientation(self.id, pos, orn)

# TODO: This should be more fleshed out in the future to get all relevant data
    def get_data(self) -> dict:
        data = {}
        data["pose"] = self.get_curr_pose()
        return data

# TODO: Create functions for keeping track of links similar to joints


class ActuatedObject(ObjectBase):
    def __init__(self, id: int = None, path: str = None, name: str = None):
        super().__init__(id=id, path=path, name=name)
        self.joint_dict = {}
        self.create_joint_dict()

    def create_joint_dict(self):
        # creates joint dictionary with style = name: number
        for i in range(p.getNumJoints(self.id)):
            info = p.getJointInfo(self.id, i)
            joint_name = info[1].decode("ascii")
            joint_type = info[2]
            joint_id = info[0]
            # filters out fixed joints
            if (joint_type != 4):
                self.joint_dict[joint_name] = joint_id
        logging.info("Joint dict: ", self.joint_dict)

    def get_joint_dict(self) -> dict:
        return self.joint_dict

    def get_joint_names(self) -> list:
        return list(self.joint_dict.keys())

    def get_joint_numbers(self) -> list:
        return list(self.joint_dict.values())

    # takes a list of names and returns the joint numbers
    def get_joints_by_name(self, names: list = []) -> list:
        joint_num_list = []
        for i in names:
            if(i in self.joint_dict):
                joint_num_list.append(self.joint_dict[i])
            else:
                logging.warn("Could not find " + str(i) + " in joint dict")
        return joint_num_list

    # gets joint names from a list of joint numbers
    # TODO: REDO THIS FUNCTION, IMPLEMENTED WEIRD
    def get_joints_by_number(self, numbers: list = []) -> list:
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
        '''
        Get the current pose angle of joints
        Stores in self.curr_joint_angle : current joint angles as a list
        :param joint_indices: List of particular joint indices to get angles for. If None, returns all joint angle values.
        '''
        curr_joint_poses = []
        if joint_numbers is None:
            curr_joint_states = p.getJointStates(
                self.id, self.joint_dict.values())
        else:
            curr_joint_states = p.getJointStates(self.id, joint_numbers)
        for joint in range(0, len(curr_joint_states)):
            curr_joint_poses.append(curr_joint_states[joint][0])
        return curr_joint_poses


# TODO: This should be more fleshed out in the future to get all relevant data

    def get_data(self) -> dict:
        data = {}
        data["pose"] = self.get_curr_pose()
        names = self.get_joint_names()
        angles = self.get_joint_angles()
        angle_dict = {}
        for i in range(len(names)):
            angle_dict[names[i]] = angles[i]
        data["joint_angles"] = angle_dict
        return data
