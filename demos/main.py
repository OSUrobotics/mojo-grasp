from matplotlib.pyplot import close
from mojograsp.simcore import record_data
import pybullet as p
import pybullet_data
import pathlib
import open_hand_phase
import close_hand_phase
import asterisk_env
from mojograsp.simcore.sim_manager import SimManagerDefault
from mojograsp.simcore.environment import EnvironmentDefault
from mojograsp.simcore.state import StateDefault
from mojograsp.simcore.reward import RewardDefault
from mojograsp.simcore.environment import EnvironmentDefault
from mojograsp.simcore.record_data import RecordDataDefault
from mojograsp.simobjects.two_finger_gripper import TwoFingerGripper
from mojograsp.simobjects.object_base import ObjectBase

# resource paths
current_path = str(pathlib.Path().resolve())
hand_path = current_path+"/2v2_nosensors/2v2_nosensors.urdf"
cube_path = current_path + "/2v2_nosensors_objects/2v2_nosensors_cuboid_small.urdf"

# start pybullet
physics_client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)
p.resetDebugVisualizerCamera(cameraDistance=.02, cameraYaw=0, cameraPitch=-89.9999,
                             cameraTargetPosition=[0, 0.1, 0.5])

# load objects
plane_id = p.loadURDF("plane.urdf")
hand_id = p.loadURDF(hand_path, useFixedBase=True,
                     basePosition=[0.0, 0.0, 0.05])
cube_id = p.loadURDF(cube_path, basePosition=[0.0, 0.17, .06])
hand = TwoFingerGripper(hand_id)
cube = ObjectBase(cube_id)


# change visual of gripper
p.changeVisualShape(hand_id, -1, rgbaColor=[0.3, 0.3, 0.3, 1])
p.changeVisualShape(hand_id, 0, rgbaColor=[1, 0.5, 0, 1])
p.changeVisualShape(hand_id, 1, rgbaColor=[0.3, 0.3, 0.3, 1])
p.changeVisualShape(hand_id, 2, rgbaColor=[1, 0.5, 0, 1])
p.changeVisualShape(hand_id, 3, rgbaColor=[0.3, 0.3, 0.3, 1])


#state and reward
state = StateDefault()
reward = RewardDefault()

#environment and recording
env = asterisk_env.AsteriskEnv(hand=hand, obj=cube)

# sim manager
manager = SimManagerDefault(num_episodes=4, env=env)

open_hand = open_hand_phase.OpenHand(hand, cube)
close_hand = close_hand_phase.CloseHand(hand, cube)
manager.add_phase("open", open_hand, start=True)
manager.add_phase("close", close_hand)

manager.run()
manager.stall()
