

class Phase:
	_sim = None
	
	def __init__(self):
		print("Phase initialized")
	
	def setup(self):
		print("Phase setting up")

	def pre_step(self):
		print("Pre step")

	def post_step(self):
		print("Post step")

	def select_action(self):
		print("Selecting action")

	def execute_action(self, action):
		print("Executing action")

	def phase_exit_condition(self, episode_step):
		print("Exit condition checked")

	def phase_complete(self):
		print("Phase complete")
