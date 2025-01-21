from mojograsp.simobjects.object_base import ObjectBase
from scipy.spatial.transform import Rotation as R

class ObjectWithVelocity(ObjectBase):
    def __init__(self, id: int = None, path: str = None, name: str = None, physicsClientId: int = 0):
        super().__init__(id=id, path=path, name=name,physicsClientId=physicsClientId)

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
        data['z_angle'] = R.from_quat(data['pose'][1]).as_euler('xyz')[-1]
        return data
    
    def get_path(self):
        """
        Returns the path of the object.

        :return: path of the object
        :rtype: str
        """
        return self.path