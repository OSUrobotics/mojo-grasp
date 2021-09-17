import mojograsp
import pybullet as p
import numpy as np
import phase1_full
import phase2_full


if __name__ == '__main__':
    # setting up simmanager/physics server
    manager = mojograsp.simmanager.SimManager_Pybullet(rl=False)
    # setting camera
    p.resetDebugVisualizerCamera(cameraDistance=.4, cameraYaw=0, cameraPitch=-45, cameraTargetPosition=[.1, 0, .1])
    hand_path = '/Users/asar/Desktop/Grimm\'s Lab/Manipulation/PyBulletStuff/mojo-grasp/hand_generation/hand_models/2v2_nosensors/2v2_nosensors.urdf'
    # hand_path = "/Users/asar/PycharmProjects/InHand-Manipulation/ExampleSimWorld-Josh/2v2_hands/999/testing.sdf"

    object_path = '/Users/asar/Desktop/Grimm\'s Lab/Manipulation/PyBulletStuff/mojo-grasp/hand_generation/object_models/2v2_nosensors/2v2_nosensors_cuboid_small.urdf'

    hand = mojograsp.hand.Hand(hand_path, fixed=True)
    cube = mojograsp.objectbase.ObjectBase(object_path, fixed=False)

    sim_env = mojograsp.environment.Environment(hand=hand, objects=cube, steps=15)
    manager.add_env(sim_env)
    open = phase1_full.OpenHand('open phase')
    close = phase2_full.CloseHand('close phase')

    manager.add_phase(open.name, open, start=True)
    manager.add_phase(close.name, close)
    # print("STATE: {}".format(open.state.update()))

    # running simulation
    manager.run()
    # stalling so it doesnt exit
    manager.stall()