import pybullet as p
from . import objectbase
from . import handgeometry

from mojograsp.simcore.actuators import fullyactuated, underactuated
from mojograsp.simcore.sensors import sensorbase

class Hand(objectbase.ObjectBase):

	def __init__(self, filename, underactuation=False, fixed=False):
		#calls initialization for objectbase class
		super().__init__(filename, fixed)

		#initialize important variables
		self.joint_index = {}
		self.sensor_index = {}
		self.create_sensor_index()
		self.create_joint_index()

		#creates instances of geometry class and actuators class
		self.geometry = handgeometry.HandGeometry(self.joint_index)
		if(underactuation == False):
			self.actuation = fullyactuated.FullyActuated(self.joint_index, self.id)
		else:
			self.actuation = underactuated.UnderActuated(self.joint_index, self.id)


	#rough draft of sensor index creation
	def create_sensor_index(self):
		print("creating sensors")
		for i in range(p.getNumJoints(self.id)):
			info = p.getJointInfo(self.id, i)
			sensor_name = info[1].decode("ascii")
			sensor_type = info[2]
			sensor_id = info[0]
			
			#if any joints have name with substring sensor we create a sensor object and add it to the index
			if(sensor_name.find("sensor") != -1):
				self.sensor_index[sensor_name] = sensorbase.SensorBase(self.id, sensor_name) 	



	#creates joint dictionary with style name: number
	def create_joint_index(self):
		print("creating joint index")
		for i in range(p.getNumJoints(self.id)):
			info = p.getJointInfo(self.id, i)
			joint_name = info[1].decode("ascii")
			joint_type = info[2]
			joint_id = info[0]
			#filters out fixed joints to exclude sensors but will need better solution in the future
			if(joint_type != 4):
				self.joint_index[joint_name] = joint_id	

		print(self.joint_index)



		
		
