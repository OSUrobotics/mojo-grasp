import pybullet as p


class SensorBase():
	#placeholder
	def __init__(self, unique_id, sensor_name):
		self.id = unique_id
		self.name = sensor_name
		self.location = None
		self.type = None
		print("Creating sensor " + str(self.name))


