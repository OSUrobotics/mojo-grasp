#!/usr/bin/env python

import pybullet as p
import time
import pybullet_data
import os

class sim_tester():

    def __init__(self, gripper_name, object_type, object_size):
        self.gripper_name = gripper_name
        self.object_type = object_type
        self.object_size = object_size

        self.directory = os.path.dirname(__file__)


    def main(self):
            
        physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
        p.setGravity(0, 0, -10)
        LinkId = []
        #planeId = p.loadURDF("sphere2red.urdf")
        cubeStartPos = [0, 0, 1]
        cubeStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
        #boxId = p.loadSDF("TestScript_objFiles.sdf")

        # file_number = '131'

        # boxId = p.loadURDF(f"{directory}/hand_models/3v2_test_hand/3v2_test_hand.urdf", useFixedBase=1)
        # boxId = p.loadURDF(f"{directory}/hand_models/testing/testing.urdf", useFixedBase=1)
        boxId = p.loadURDF(f"{self.directory}/hand_models/{self.gripper_name}/{self.gripper_name}.urdf", useFixedBase=1)

        gripper = boxId

        obj = p.loadURDF(f"{self.directory}/object_models/{self.gripper_name}/{self.gripper_name}_{self.object_type}_{self.object_size}.urdf", useFixedBase=0, basePosition=[0, 0.2, 0])

        p.resetDebugVisualizerCamera(cameraDistance=.2, cameraYaw=180, cameraPitch=-91, cameraTargetPosition=[0, 0.1, 0.1])
        # p.resetDebugVisualizerCamera(cameraDistance=.2, cameraYaw=90, cameraPitch=0, cameraTargetPosition=[.1, 0, .1])
        #print(p.getNumJoints(gripper))
        #print(p.getJointInfo(gripper, 0)[12].decode("ascii"))

        for i in range(0, p.getNumJoints(gripper)):
            # if i == 0:
            #     LinkId.append(0)
            #     continue 
            p.setJointMotorControl2(gripper, i, p.POSITION_CONTROL, targetPosition=0, force=0)
            linkName = p.getJointInfo(gripper, i)[12].decode("ascii")
            if "sensor" in linkName:
                LinkId.append("skip")
            else:
                LinkId.append(p.addUserDebugParameter(linkName, -3.14, 3.14, 0))



        while p.isConnected():

            p.stepSimulation()
            time.sleep(1. / 240.)

            for i in range(0, len(LinkId)):
                if LinkId[i] != "skip":
                    linkPos = p.readUserDebugParameter(LinkId[i])
                    p.setJointMotorControl2(gripper, i, p.POSITION_CONTROL, targetPosition=linkPos)


        p.disconnect()

if __name__ == '__main__':

    gripper_name = input("Name of the gipper (ex. 3v2_test_hand):  ")
    if gripper_name == "j":
        gripper_name = "testing1"
        object_type = "cuboid"
        object_size = "small"
    else:
        object_type = input("Object type(currently just cuboid):  ")
        object_size = input("Object size (small medium or large):  ")

    sim_test = sim_tester(gripper_name, object_type, object_size)
    sim_test.main()