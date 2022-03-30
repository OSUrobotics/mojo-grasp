from mojograsp.simcore import record_data
import pybullet as p
import pybullet_data
import pathlib
from mojograsp.simcore.sim_manager import SimManagerDefault
from mojograsp.simcore.environment import EnvironmentDefault
from mojograsp.simcore.state import StateBlank
from mojograsp.simcore.reward import RewardBlank
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
hand = Hand(hand_path, fixed=True, base_pos=[0.0, 0.0, 0.08])
cube = ObjectBase(object_path, fixed=False, base_pos=[0.0, 0.17, 0])

#state and reward
state = StateBlank()
reward = RewardBlank()

#environment and recording
env = EnvironmentDefault(hand=hand, object=object, state=state, reward=reward)
rec = RecordDataDefault(data_path=current_path)


manager = SimManagerDefault(env=env, record=rec)
manager.run()
