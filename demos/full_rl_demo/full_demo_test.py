import mojograsp
import pybullet as p
import numpy as np
import phase1_full
import phase2_full
import phase4_full_rl
import pathlib
import user_functions_test



if __name__ == '__main__':
    """
    Part 1: Set up
    """
    # setting up simmanager
    current_path = str(pathlib.Path().resolve())
    #ENTER REPLAY BUFFER FILE PATH HERE, ex: cube_all_episodes.csv in data directory
    replay_buffer_episode_file = None
    manager = mojograsp.simmanager.SimManagerPybullet(num_episodes=10, rl=False, data_directory_path=current_path+"/data",
              replay_episode_file=replay_buffer_episode_file, user_func=user_functions_test.UserFunctionsTest())

    # setting camera
    p.resetDebugVisualizerCamera(cameraDistance=.02, cameraYaw=0, cameraPitch=-89.9999,
                                 cameraTargetPosition=[0, 0.1, 0.5])

    # Instantiating sim objects
    hand_path = current_path+"/hand_generation/hand_models/2v2_nosensors/2v2_nosensors.urdf"
    object_path =current_path+"/hand_generation/object_models/2v2_nosensors/2v2_nosensors_cuboid_small.urdf"
    hand = mojograsp.hand.Hand(hand_path, fixed=True, base_pos=[0.0, 0.0, 0.08])
    cube = mojograsp.objectbase.ObjectBase(object_path, fixed=False, base_pos=[0.0, 0.17, 0])

    train = ['g', 'h']
    test = ['b_c'] #, 'c_d', 'f_g', 'g_h', 'h_a']

    # Instantiating environment
    # sim_env = mojograsp.environment.Environment(hand=hand, objects=cube, steps=2, directions=['a_b', 'b_c'],
    #                                             subjects=['sub1', 'sub2', 'sub3'], trials=[1, 2, 3, 4, 5])
    sim_env = mojograsp.environment.Environment(hand=hand, objects=cube, steps=2, directions=test,
                                                subjects=['sub1', 'sub2', 'sub3'], trials=[1, 2, 3, 4, 5],
                                                trial_types=['human', 'expected'])

    # Adding environment
    manager.add_env(sim_env)

    """
    Part 2: The Phases
    """
    # Instantiating phases
    open = phase1_full.OpenHand('open phase')
    close = phase2_full.CloseHand('close phase')
    move_rl = phase4_full_rl.MoveRL('move rl')

    # Adding phases
    manager.add_phase(open.name, open, start=True)
    manager.add_phase(close.name, close)
    manager.add_phase(move_rl.name, move_rl)

    # running simulation
    manager.run()
    # stalling so it doesnt exit
    manager.stall()
