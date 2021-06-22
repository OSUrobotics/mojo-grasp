import pybullet as p

class ObjectBase():

	def __init__(self, filename, fixed):
		print("init objectbase")

		#contains base variables for all objects
		self.model_file = filename
		self.id = None
		self.origin = None
		self.position = None
		self.orientation = None
		self.obj_id = None
		self.fixed = fixed
		self.load_object()

	#loads object into pybullet, needs to be changed to support urdf, sdf and others right now only supports urdf with no arguments other than fixed
	def load_object(self):
		# self.id = p.loadURDF(self.model_file, useFixedBase=self.fixed)
		ids = p.loadSDF(self.model_file)
		self.id = ids[0]
		self.obj_id = ids[1]
		print("loading objects")

