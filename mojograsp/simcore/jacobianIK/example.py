import jacobian_IK
import matrix_helper
from copy import deepcopy
import pybullet as p
import pybullet_data
import pathlib


def setup_sim():
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -10)
    plane_id = p.loadURDF("plane.urdf", flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)


def setup_hand(hand_path, hand_info):
    # load urdf
    hand_id = p.loadURDF(
        hand_path, useFixedBase=True, basePosition=[0.0, 0.0, 0.05],
        flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES | p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS)

    # IMPORTANT IK SETUP STEP HERE
    ik_f1 = jacobian_IK.JacobianIK(hand_id, deepcopy(hand_info["finger1"]))
    ik_f2 = jacobian_IK.JacobianIK(hand_id, deepcopy(hand_info["finger2"]))
    distal_f1_index = ik_f1.finger_fk.link_ids[-1]
    distal_f2_index = ik_f2.finger_fk.link_ids[-1]
    proximal_f1_index = ik_f1.finger_fk.link_ids[0]
    proximal_f2_index = ik_f2.finger_fk.link_ids[0]

    # get joints in starting positions, this is arbitrary, just have all the joints start with at least a slight angle
    p.resetJointState(hand_id, proximal_f1_index, -.5)
    p.resetJointState(hand_id, proximal_f2_index, .5)
    p.resetJointState(hand_id, distal_f1_index, .4)
    p.resetJointState(hand_id, distal_f2_index, -.4)
    # UPDATE THE FK FROM SIM, VERY IMPORTANT THAT YOU CALL AFTER RESETING THE JOINT STATE
    ik_f1.finger_fk.update_angles_from_sim()
    ik_f2.finger_fk.update_angles_from_sim()

    return hand_id, ik_f1, ik_f2, distal_f1_index, distal_f2_index


# Example hand dimensions, yes the naming convention is horrendous for fingers, I need to go back and change old code to use better naming.
# I believe this is the base 2v2 dimensions
hand_info = {"finger1": {"name": "finger0", "num_links": 2, "link_lengths": [[0, .072, 0], [0, .072, 0]]},
             "finger2": {"name": "finger1", "num_links": 2, "link_lengths": [[0, .072, 0], [0, .072, 0]]}}


# setup sim
setup_sim()
current_path = str(pathlib.Path().resolve())
hand_path = current_path+"/2v2_Demo/hand/2v2_Demo.urdf"
# How I set up the hand with the IK in sim
hand_id, ik_f1, ik_f2, distal_f1, distal_f2 = setup_hand(hand_path, hand_info)

# EXTRA DEMO STUFF
print("INITIAL EE POINT F1: ", ik_f1.finger_fk.calculate_forward_kinematics())
print("INITIAL EE POINT F2: ", ik_f2.finger_fk.calculate_forward_kinematics())
# Changing EE position
ik_f1.finger_fk.update_ee_end_point([0, .05, 1])
ik_f2.finger_fk.update_ee_end_point([0, .05, 1])
# This should only be done if you want to change ee location at the beginning of simulation, if you want to have a changing
# contact point with the IK solver use the optional argument in calculate_ik
print("CHANGED EE POINT F1: ", ik_f1.finger_fk.calculate_forward_kinematics())
print("CHANGED EE POINT F2: ", ik_f2.finger_fk.calculate_forward_kinematics())
# resetting back EE position
ik_f1.finger_fk.update_ee_end_point([0, .072, 1])
ik_f2.finger_fk.update_ee_end_point([0, .072, 1])


# found is whether we took at least a step towards goal, it is the number of iterations we took, angles are joint angles
print("IK SOLVE TO TARGET POSITIONS")
found1, angles_f1, it1 = ik_f1.calculate_ik(target=[.03, .135], ee_location=None)
found2, angles_f2, it2 = ik_f2.calculate_ik(target=[-.03, .135], ee_location=None)
print("JOINT ANGLES FOR F1, TARGET [.03, .135]: ", angles_f1)
print("JOINT ANGLES FOR F2, TARGET [-.03, .135]: ", angles_f2)

# moving to new position
p.resetJointState(hand_id, 0, angles_f1[0])
p.resetJointState(hand_id, 1, angles_f1[1])
p.resetJointState(hand_id, 2, angles_f2[0])
p.resetJointState(hand_id, 3, angles_f2[1])

# You dont need to call this if you are calling the calculate_ik at every timestep, it will handle it for you after the sim steps
# This is just for the example
ik_f1.finger_fk.update_angles_from_sim()
ik_f2.finger_fk.update_angles_from_sim()
p.stepSimulation()
print("EE AFTER IK AT TARGET F1: ", ik_f1.finger_fk.calculate_forward_kinematics())
print("EE AFTER IK AT TARGET F2: ", ik_f2.finger_fk.calculate_forward_kinematics())
