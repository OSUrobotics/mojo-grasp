import pybullet as p


class SensorBase():
    """
    SensorBase Class will eventually be the base class that all other supported sensors inherit from.
    The idea is to be able to attach and sensor to any object easily. 
    """

    def __init__(self, unique_id, sensor_name):
        """Placeholder"""
        self.id = unique_id
        self.name = sensor_name
        self.location = None
        self.type = None
        print("Creating sensor " + str(self.name))
