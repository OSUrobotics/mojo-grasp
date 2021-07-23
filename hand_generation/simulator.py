#!/usr/bin/env python

import pybullet as p
import time
import pybullet_data
import os


physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
p.setGravity(0, 0, -10)
LinkId = []
#planeId = p.loadURDF("sphere2red.urdf")
cubeStartPos = [0, 0, 1]
cubeStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
#boxId = p.loadSDF("TestScript_objFiles.sdf")

file_number = '131'

directory = os.path.dirname(__file__)

# boxId = p.loadURDF(f"{directory}/hand_models/3v2_test_hand/3v2_test_hand.urdf", useFixedBase=1)
boxId = p.loadURDF(f"{directory}/hand_models/testing/testing.urdf", useFixedBase=1)


gripper = boxId

obj = p.loadURDF(f"{directory}/object_models/3v2_test_hand/3v2_test_hand_cuboid_small.urdf", useFixedBase=1, basePosition=[0, 0.2, 0])

p.resetDebugVisualizerCamera(cameraDistance=.2, cameraYaw=90, cameraPitch=0, cameraTargetPosition=[.1, 0, .1])
#print(p.getNumJoints(gripper))
#print(p.getJointInfo(gripper, 0)[12].decode("ascii"))

for i in range(0, p.getNumJoints(gripper)):
    # if i == 0:
    #     LinkId.append(0)
    #     continue
    p.setJointMotorControl2(gripper, i, p.POSITION_CONTROL, targetPosition=0, force=0)
    linkName = p.getJointInfo(gripper, i)[12].decode("ascii")
    LinkId.append(p.addUserDebugParameter(linkName, -3.14, 3.14, 0))



while p.isConnected():

    p.stepSimulation()
    time.sleep(1. / 240.)

    for i in range(0, p.getNumJoints(gripper)):
        linkPos = p.readUserDebugParameter((LinkId[i]))
        p.setJointMotorControl2(gripper, i, p.POSITION_CONTROL, targetPosition=linkPos)


p.disconnect()
