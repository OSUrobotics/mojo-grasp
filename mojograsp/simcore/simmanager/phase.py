

class Phase():
	
	def __init__(self, simobjects):
		print("Phase initialized")
		None
	
	def setup(self):
		print("Phase setting up")
		None

	def pre_step(self):
		print("Pre step")
		None

	def post_step(self):
		print("Post step")
		None

	def select_action(self):
		print("Selecting action")
		return None

	def execute_action(self):
		print("Executing action")
		None

	def phase_exit_condition(self, episode_step):
		print("Exit condition checked")
		return True

	def phase_complete(self):
		print("Phase complete")
		None
