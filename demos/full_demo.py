import mojograsp
import pybullet as p
import numpy as np
import phase1_full
import phase2_full
import phase3_full
import phase4_full_rl


if __name__ == '__main__':
    """
    Part 1: Set up
    """
    # setting up simmanager
    manager = mojograsp.simmanager.SimManagerPybullet(num_episodes=2, rl=False)

    # setting camera
    p.resetDebugVisualizerCamera(cameraDistance=.02, cameraYaw=0, cameraPitch=-89.9,
                                 cameraTargetPosition=[0, 0.1, 0.5])
    # p.resetDebugVisualizerCamera(cameraDistance=2, cameraYaw=0, cameraPitch=90.0,
    #                              cameraTargetPosition=[0, 0, 10])
    # p.resetDebugVisualizerCamera(cameraDistance=2, cameraYaw=90, cameraPitch=-90.0,
    #                              cameraTargetPosition=[0, 0, -10])

    # Instantiating sim objects
    hand_path = "/Users/asar/PycharmProjects/InHand-Manipulation/ExampleSimWorld-Josh/2v2_nosensors_hand" \
                "/2v2_nosensors.urdf"
    object_path = "/Users/asar/PycharmProjects/InHand-Manipulation/ExampleSimWorld-Josh/2v2_nosensors_hand_object/" \
                  "2v2_nosensors_cuboid_small.urdf"
    hand = mojograsp.hand.Hand(hand_path, fixed=True, base_pos=[0.0, 0.0, 0.08])
    cube = mojograsp.objectbase.ObjectBase(object_path, fixed=False, base_pos=[0.0, 0.17, 0])

    # Instantiating environment
    sim_env = mojograsp.environment.Environment(hand=hand, objects=cube, steps=2)

    # Adding environment
    manager.add_env(sim_env)

    """
    Part 2: The Phases
    """
    # Instantiating phases
    open = phase1_full.OpenHand('open phase')
    close = phase2_full.CloseHand('close phase')
    # move_rl = phase3_full.MoveHand('move rl')
    move_rl = phase4_full_rl.MoveRL('move rl')

    # Adding phases
    manager.add_phase(open.name, open, start=True)
    manager.add_phase(close.name, close)
    manager.add_phase(move_rl.name, move_rl)

    # running simulation
    manager.run()
    # stalling so it doesnt exit
    manager.stall()
