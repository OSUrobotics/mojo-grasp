import pybullet as p
from . import objectbase
from . import handgeometry

from mojograsp.simcore.actuators import fullyactuated, underactuated

class Hand(objectbase.ObjectBase):

	def __init__(self, filename, underactuation=False):
		#calls initialization for objectbase class
		super().__init__(filename)

		joint_index = {}
		sensor_index = {}
		self.create_sensor_index()
		self.create_joint_index()

		#creates instances of geometry class and actuators class
		self.geometry = handgeometry.HandGeometry(joint_index)
		if(underactuation == False):
			self.actuation = fullyactuated.FullyActuated(joint_index)
		else:
			self.actuation = underactuated.UnderActuated(joint_index)


		
	def create_sensor_index(self):
		print("creating sensors")


	def create_joint_index(self):
		print("creating joint index")
		
		
