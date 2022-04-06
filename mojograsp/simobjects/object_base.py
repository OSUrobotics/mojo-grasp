import pybullet as p
import logging


class ObjectBase:

    def __init__(self, id: int = None):
        # all coordinate frame magic will be done here.
        logging.info("Object Created with id: {}".format(id))
        self.id = id

    def get_curr_pose(self) -> list[float]:
        pose = []
        curr_pos, curr_orn = p.getBasePositionAndOrientation(self.id)
        pose.append(curr_pos)
        pose.append(curr_orn)
        return pose

    def get_dimensions(self) -> list[float]:
        return p.getVisualShapeData(self.id)[0][3]

    def set_curr_pose(self, pos, orn):
        if len(orn) == 3:
            orn = p.getQuaternionFromEuler(orn)
        p.resetBasePositionAndOrientation(self.id, pos, orn)


# TODO: Create functions for keeping track of links similar to joints
class ActuatedObject(ObjectBase):
    def __init__(self, id: int = None):
        super().__init__(id=id)
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

    # gets joint names from index
    def get_joint_names(self) -> list[str]:
        return list(self.joint_dict.keys())

        # gets joint numbers from index
    def get_joint_numbers(self) -> list[int]:
        return list(self.joint_dict.values())

    # takes a list of names and returns the joint numbers
    def get_joints_by_name(self, names: list[str] = []) -> list[int]:
        joint_num_list = []
        for i in names:
            if(i in self.joint_index):
                joint_num_list.append(self.joint_index[i])
            else:
                logging.warn("Could not find " + str(i) + " in joint index")
        return joint_num_list

    # gets joint names from a list of joint numbers
    # TODO: REDO THIS FUNCTION, IMPLEMENTED WEIRD
    def get_joints_by_number(self, numbers: list[int] = []) -> list[str]:
        joint_name_list = []
        name = list(self.joint_index.keys())
        num = list(self.joint_index.values())
        for i in numbers:
            found = False
            for j in range(len(num)):
                if(i == num[j]):
                    joint_name_list.append(name[j])
                    found = True
                    break
            if(found == False):
                logging.warn("Could not find joint " +
                             str(i) + " in joint index")
        return joint_name_list
