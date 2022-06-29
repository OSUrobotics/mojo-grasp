import pybullet as p
import pybullet_data
import pathlib
import manipulation_phase
import expert_env
import expert_action
import expert_reward
import pandas as pd
from mojograsp.simcore.sim_manager import SimManagerDefault
from mojograsp.simcore.state import StateDefault
from mojograsp.simcore.reward import RewardDefault
from mojograsp.simcore.record_data import RecordDataJSON
from mojograsp.simobjects.two_finger_gripper import TwoFingerGripper
from mojograsp.simobjects.object_base import ObjectBase


# resource paths
current_path = str(pathlib.Path().resolve())
hand_path = current_path+"/resources/2v2_nosensors/2v2_nosensors.urdf"
cube_path = current_path + \
    "/resources/object_models/2v2_mod/2v2_mod_cuboid_small.urdf"
data_path = current_path+"/data/"
points_path = current_path+"/resources/points.csv"

# Load in the cube goal positions
df = pd.read_csv(points_path, index_col=False)
x = df["x"]
y = df["y"]

# start pybullet
physics_client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)
p.resetDebugVisualizerCamera(cameraDistance=.02, cameraYaw=0, cameraPitch=-89.9999,
                             cameraTargetPosition=[0, 0.1, 0.5])

# load objects into pybullet
plane_id = p.loadURDF("plane.urdf")
hand_id = p.loadURDF(hand_path, useFixedBase=True,
                     basePosition=[0.0, 0.0, 0.05])
cube_id = p.loadURDF(cube_path, basePosition=[0.0, 0.16, .05])

# Create TwoFingerGripper Object and set the initial joint positions
hand = TwoFingerGripper(hand_id, path=hand_path)

p.resetJointState(hand_id, 0, .75)
p.resetJointState(hand_id, 1, -1.4)
p.resetJointState(hand_id, 2, -.75)
p.resetJointState(hand_id, 3, 1.4)

# Create ObjectBase for the cube object
cube = ObjectBase(cube_id, path=cube_path)

# change visual of gripper
p.changeVisualShape(hand_id, -1, rgbaColor=[0.3, 0.3, 0.3, 1])
p.changeVisualShape(hand_id, 0, rgbaColor=[1, 0.5, 0, 1])
p.changeVisualShape(hand_id, 1, rgbaColor=[0.3, 0.3, 0.3, 1])
p.changeVisualShape(hand_id, 2, rgbaColor=[1, 0.5, 0, 1])
p.changeVisualShape(hand_id, 3, rgbaColor=[0.3, 0.3, 0.3, 1])
# p.setTimeStep(1/2400)

# state and reward
state = StateDefault(objects=[hand, cube])
action = expert_action.ExpertAction()
reward = expert_reward.ExpertReward()

# data recording
record_data = RecordDataJSON(
    data_path=data_path, state=state, action=action, reward=reward, save_all=True)

# environment and recording
env = expert_env.ExpertEnv(hand=hand, obj=cube)

# sim manager
manager = SimManagerDefault(num_episodes=len(x), env=env, record_data=record_data)

# Create phase and pass it to the sim manager
manipulation = manipulation_phase.Manipulation(
    hand, cube, x, y, state, action, reward)
manager.add_phase("manipulation", manipulation, start=True)

# Run the sim
manager.run()
manager.stall()
