from mojograsp.simobjects.object_base import ObjectBase

class ObjectWithVelocity(ObjectBase):
    def __init__(self, id: int = None, path: str = None, name: str = None):
        super().__init__(id=id, path=path, name=name)

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