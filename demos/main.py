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
from mojograsp.simobjects.hand import Hand
from mojograsp.simobjects.objectbase import ObjectBase

# resource paths
current_path = str(pathlib.Path().resolve())
hand_path = current_path+"/2v2_nosensors/2v2_nosensors.urdf"
object_path = current_path + "/2v2_nosensors_objects/2v2_nosensors_cuboid_small.urdf"


# start pybullet
physics_client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)
p.resetDebugVisualizerCamera(cameraDistance=.02, cameraYaw=0, cameraPitch=-89.9999,
                             cameraTargetPosition=[0, 0.1, 0.5])
plane_id = p.loadURDF("plane.urdf")

# objects
hand = Hand(hand_path, fixed=True, base_pos=[0.0, 0.0, 0.05])
cube = ObjectBase(object_path, fixed=False, base_pos=[0.0, 0.17, .06])

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
