import pybullet as p


class FullyActuated():
	def __init__(self, joint_index):
		self.joint_index = joint_index
		
	def set_joints(self):
		print("setting joints")
