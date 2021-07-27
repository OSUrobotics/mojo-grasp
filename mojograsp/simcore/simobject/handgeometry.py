import pybullet as p
from . import objectbase

class HandGeometry(objectbase.ObjectBase):

	def __init__(self, joint_index):
		self.joint_index = joint_index
		self.sdf = None

	def create_sdf(self):
		print("creating sdf")
		



