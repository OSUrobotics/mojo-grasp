import pybullet as p


from mojograsp.simobjects.two_finger_gripper import TwoFingerGripper
from pybullet_utils.bullet_client import BulletClient
from mojograsp.simcore.jacobianIK.jacobian_IK import JacobianIK
from copy import deepcopy
import numpy as np

# TODO: Make convenience functions for end effectors
class IKGripper(TwoFingerGripper):
    """TwoFingerGripper Class is a child class of ActuatedObject"""

    def __init__(self, id: int = None, path: str = None, name: str = "two_finger_gripper", physicsClientId:BulletClient=None):
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
        
        if 'Hand_A' in path:
            hand_info = {"finger1": {"name": "finger0", "num_links": 2, "link_lengths": [[0, .072, 0], [0, .072, 0]]},
                         "finger2": {"name": "finger1", "num_links": 2, "link_lengths": [[0, .072, 0], [0, .072, 0]]}}
        elif 'Hand_B' in path:
            hand_info = {"finger1": {"name": "finger0", "num_links": 2, "link_lengths": [[0, .0936, 0], [0, .0504, 0]]},
                         "finger2": {"name": "finger1", "num_links": 2, "link_lengths": [[0, .0936, 0], [0, .0504, 0]]}}
        else:
            raise TypeError('unrecognized hand type, add link lengths to simobjects/IK_gripper.py')
        self.ik_f1 = JacobianIK(id,deepcopy(hand_info['finger1']))
        
        self.ik_f2 = JacobianIK(id,deepcopy(hand_info['finger2']))

    def calculate_ik(self, new_finger_poses):
        _, finger_1_angs_kegan, _ = self.ik_f1.calculate_ik(target=new_finger_poses[:2], ee_location=None)
        _, finger_2_angs_kegan, _ = self.ik_f2.calculate_ik(target=new_finger_poses[2:], ee_location=None)
        return finger_1_angs_kegan, finger_2_angs_kegan
    
    def calculate_fk(self, angles):
        self.ik_f1.finger_fk.set_joint_angles(angles[0:2])
        f1_pos = self.ik_f1.finger_fk.calculate_forward_kinematics()
        self.ik_f2.finger_fk.set_joint_angles(angles[2:4])
        f2_pos = self.ik_f2.finger_fk.calculate_forward_kinematics()
        return f1_pos, f2_pos
    
    def update_angles_from_sim(self):
        self.ik_f1.finger_fk.update_angles_from_sim()
        self.ik_f2.finger_fk.update_angles_from_sim()

    def get_data(self) -> dict:
        data_dict = super().get_data()
        
        self.update_angles_from_sim()

        f1_jacobian = self.ik_f1.calculate_jacobian()
        f2_jacobian = self.ik_f2.calculate_jacobian()

        f1_eigenvalues, f1_eigenvectors = np.linalg.eig(f1_jacobian)
        f2_eigenvalues, f2_eigenvectors = np.linalg.eig(f2_jacobian)
        data_dict['eigenvalues'] = [f1_eigenvalues[0],f1_eigenvalues[1],f2_eigenvalues[0],f2_eigenvalues[1]]
        data_dict['eigenvectors'] = [f1_eigenvectors[0,0],f1_eigenvectors[0,1],f1_eigenvectors[1,0],f1_eigenvectors[1,1],
                                     f2_eigenvectors[0,0],f2_eigenvectors[0,1],f2_eigenvectors[1,0],f2_eigenvectors[1,1]]

        return data_dict