import pybullet as p
import pybullet_data
import pathlib

# Custom Phases
import open_hand_phase
import close_hand_phase

# Custom Environmnet
import basic_env

# Mojograsp Defaults
from mojograsp.simcore.sim_manager import SimManagerDefault
from mojograsp.simcore.state import StateDefault
from mojograsp.simcore.record_data import RecordDataJSON
from mojograsp.simobjects.two_finger_gripper import TwoFingerGripper
from mojograsp.simobjects.object_base import ObjectBase

# Resource paths
current_path = str(pathlib.Path().resolve())
hand_path = current_path+"/resources/2v2_nosensors/2v2_nosensors.urdf"
cube_path = current_path + \
    "/resources/2v2_nosensors_objects/2v2_nosensors_cuboid_small.urdf"
data_path = current_path+"/data/"

# Start pybullet
physics_client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
# Set environment parameters
p.setGravity(0, 0, -10)
p.resetDebugVisualizerCamera(cameraDistance=.02, cameraYaw=0, cameraPitch=-89.9999,
                             cameraTargetPosition=[0, 0.1, 0.5])

# load plane object
plane_id = p.loadURDF("plane.urdf")
# load hand and cube object into pybullet environment
hand_id = p.loadURDF(hand_path, useFixedBase=True,
                     basePosition=[0.0, 0.0, 0.05])
cube_id = p.loadURDF(cube_path, basePosition=[0.0, 0.17, .06])

# Create hand and Cube Mojograsp objects by passing in paths and pybullet id's
# This allows us to pass them easier and take advantage of helper functions
hand = TwoFingerGripper(hand_id, path=hand_path)
cube = ObjectBase(cube_id, path=cube_path)


# Change visual of gripper to make it look nicer (OPTIONAL)
p.changeVisualShape(hand_id, -1, rgbaColor=[0.3, 0.3, 0.3, 1])
p.changeVisualShape(hand_id, 0, rgbaColor=[1, 0.5, 0, 1])
p.changeVisualShape(hand_id, 1, rgbaColor=[0.3, 0.3, 0.3, 1])
p.changeVisualShape(hand_id, 2, rgbaColor=[1, 0.5, 0, 1])
p.changeVisualShape(hand_id, 3, rgbaColor=[0.3, 0.3, 0.3, 1])


# We are running multiple trials and want to reset the simulation each time so we are using a custom environment
# Class that inherits from the ABC Environment class. We pass in the hand and cube object so we can reset them in
# Asterisk Env
env = basic_env.BasicEnv(hand=hand, obj=cube)

# Create the Sim Manager
manager = SimManagerDefault(num_episodes=3, env=env)

# Instantiate our phases, and pass in our objects
open_hand = open_hand_phase.OpenHand(hand, cube)
close_hand = close_hand_phase.CloseHand(hand, cube)

# Add the phases to the simmanager
# Each episode will consist of: open -> close -> STOP
manager.add_phase("open", open_hand, start=True)
manager.add_phase("close", close_hand)

# run the sim
manager.run()
manager.stall()
