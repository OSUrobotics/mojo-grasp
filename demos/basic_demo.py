import mojograsp
import pybullet as p

#example of a phase, closes the hand
class CloseHand(mojograsp.phase.Phase):

	def __init__(self, hand):
		self.hand = hand
		self.joint_nums = None
		
	def setup(self):
		print("Phase1 setup")
		self.joint_nums = self.hand.actuation.get_joint_index_numbers()
		print("Phase1 executing")

	def execute_action(self):
		tp = [-.2, -.2, .2, .2, .2]
		f = [.02, .02, .02, .02, .02]
		p.setJointMotorControlArray(self.hand.id, self.joint_nums, controlMode=p.POSITION_CONTROL,targetPositions=tp, forces=f) 

	def phase_exit_condition(self, episode_step):
		if(episode_step >= 500):
			print("Phase1 finished")
			return True
		else:
			return False

	def phase_complete(self):
		print("Beggining Phase2 open")
		return "open"

#example 2 of a phase, opens the hand
#each phase ovverides the needed functions from the parent class which simmanagers calls
class OpenHand(mojograsp.phase.Phase):

	def __init__(self, hand):
		self.hand = hand
		self.joint_nums = None
		
	def setup(self):
		print("Phase2 setup")
		self.joint_nums = self.hand.actuation.get_joint_index_numbers()
		print("Phase2 executing")

	def execute_action(self):
		tp = [0,0,0,0,0]
		f = [.02, .02, .02, .02, .02]
		p.setJointMotorControlArray(self.hand.id, self.joint_nums, controlMode=p.POSITION_CONTROL,targetPositions=tp, forces=f) 

	def phase_exit_condition(self, episode_step):
		if(episode_step >= 1000):
			print("Phase2 finished")
			return True
		else:
			return False
		
	def phase_complete(self):
		print("No Phases remaining, Episode complete")
		return None


#BASE CODE

#setting up simmanager/physics server
manager = mojograsp.simmanager.SimManager_Pybullet()
#setting camera
p.resetDebugVisualizerCamera(cameraDistance=.4, cameraYaw=0, cameraPitch=-45, cameraTargetPosition=[.1, 0, .1])
#loading our hand object using the custom class
hand = mojograsp.hand.Hand("example_hand/3v2_test_hand.urdf", fixed=True)

#creating phase1 and phase2 from above 
phase1 = CloseHand(hand)
phase2 = OpenHand(hand)
#adding our phases to the simmanager
manager.add_phase("close", phase1, start=True)
manager.add_phase("open", phase2)

#running simulation
manager.run()
#stalling so it doesnt exit
manager.stall()



