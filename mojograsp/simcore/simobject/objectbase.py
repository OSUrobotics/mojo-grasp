import pybullet as p

class ObjectBase():

	def __init__(self, filename):
		print("init objectbase")

		self.model_file = None
		self.origin = None
		self.position = None
		self.orientation = None
		self.fixed = False

	def load_object(self):
		print("loading object")

