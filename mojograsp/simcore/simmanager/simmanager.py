import time
import pybullet as p
from . import episode

class SimManager():

	def __init__(self, num_episodes=1, sim_timestep=(1. / 240.), episode_timestep=1, episode_configuration=None, gym=False):
		#initializes phase dictionary and other variables we will need
		self.phase_dict = {}
		self.current_phase = None
		self.starting_phase = None
		self.gym = gym

		#sets episode configuration object and checks if it is none, if it is we create our own empty one
		#need this for stepping later
		self.episode_configuration = episode_configuration
		if(self.episode_configuration == None):
			self.episode_configuration = episode.Episode()
			
		#variables to keep track of episodes and timestep lengths
		self.num_episodes = num_episodes
		self.episode_timestep = episode_timestep
		self.sim_timestep = sim_timestep
		
		#physics server setup, in the future needs arguments
		self.setup()


	#physics server setup, in the future needs arguments
	def setup(self):
		self.physics_client = p.connect(p.GUI)
		p.setGravity(0,0,-10)	
	
	#Prevents pybullet from closing
	def stall(self):
		while p.isConnected():
			time.sleep(1)

	#adds a phase to our phase dictionary
	def add_phase(self, phase_name, phase_object, start=False):
		self.phase_dict[phase_name] = phase_object

		#if start flag set or only phase given we set it as starting phase
		if(len(self.phase_dict) == 1 or start == True):
			self.starting_phase = phase_object

		print("Added Phase")
		
	#rough outline of simstep where phase functions called and pybullet stepped n times
	def step(self):
		#take episode pre step action
		self.episode_configuration.episode_pre_step()
		#select action before episode step
		action = self.current_phase.select_action()

		#simulator timesteps equaling one episode timestep
		for i in range(self.episode_timestep):
			#phase prestep and execute called
			self.current_phase.pre_step()
			self.current_phase.execute_action()

			#Pybullet stepped
			p.stepSimulation()
			time.sleep(self.sim_timestep)

			#phase post step called
			self.current_phase.post_step()

		#after 1 episode step we call the episode post step function
		self.episode_configuration.episode_post_step()


	#Rough outline of run, runs for set amount of episodes where within each episode all phases are run until their exit condition is met
	def run(self):
		print("RUNNING")
		
		#resets episode settings, runs episode setup and sets the current phase
		for i in range(self.num_episodes):
			self.episode_configuration.reset()	
			self.episode_configuration.setup()	
			self.current_phase = self.starting_phase
			
			#for every phase in the dictionary we step until the exit condition is met
			for j in range(len(self.phase_dict)):
				self.current_phase.setup()
				step_count = 0
				
				#while exit condition is not met call step 
				while(self.current_phase.phase_exit_condition(step_count) == False):
					self.step()
					step_count+=1
				
				#after exit condition is met we get the next phase name and set current phase to the specified value
				next_phase = self.current_phase.phase_complete()	
				if(next_phase != None):
					try:
						self.current_phase = self.phase_dict[next_phase]
					except:
						print("Error: Could not find next phase " + str(next_phase))
				else:
					break
					
				
			
		
		
