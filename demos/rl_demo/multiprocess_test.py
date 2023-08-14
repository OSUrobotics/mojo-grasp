import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

from pybullet_utils import bullet_client
from mojograsp.simcore.gym_run import run_pybullet


import time

useGUI = False
timeStep = 1./60.

# Importing the libraries
import os
import time
import multiprocessing as mp
from multiprocessing import Process, Pipe
import pybullet_data
from demos.rl_demo import rl_env
from demos.rl_demo import manipulation_phase_rl
# import rl_env
from demos.rl_demo.rl_state import StateRL, GoalHolder, RandomGoalHolder
from demos.rl_demo import rl_action
from demos.rl_demo import rl_reward
from demos.rl_demo import rl_gym_wrapper
from mojograsp.simcore.record_data import RecordDataJSON, RecordDataPKL,  RecordDataRLPKL
from mojograsp.simobjects.two_finger_gripper import TwoFingerGripper
from mojograsp.simobjects.object_with_velocity import ObjectWithVelocity
from mojograsp.simcore.priority_replay_buffer import ReplayBufferPriority
import pickle as pkl
import json
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.evaluation import evaluate_policy
import wandb
import numpy as np
import os

_RESET = 1
_CLOSE = 2
_EXPLORE = 3


def ExploreWorker(rank, num_processes, childPipe, args, poses):
    print("hi:",rank, " out of ", num_processes)  
    import pybullet as op1
    import pybullet_data as pd
    logName=""
    p1=0
    n = 0
    space = 2 
    simulations=[]
    sims_per_worker = 10

    offsetY = rank*space
    while True:
        n += 1
        try:
            # Only block for short times to have keyboard exceptions be raised.
            if not childPipe.poll(0.0001):
                continue
            message, payload = childPipe.recv()
        except (EOFError, KeyboardInterrupt):
            break
        if message == _RESET:

            p1 = bullet_client.BulletClient(op1.DIRECT)
            p1.setAdditionalSearchPath(pybullet_data.getDataPath())
            p1.setGravity(0, 0, -10)
            p1.setPhysicsEngineParameter(contactBreakingThreshold=.001)
            p1.resetDebugVisualizerCamera(cameraDistance=.02, cameraYaw=0, cameraPitch=-89.9999,
                                        cameraTargetPosition=[0, 0.1, 0.5])            
                # load objects into pybullet
            plane_id = p1.loadURDF("plane.urdf", flags=p1.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
            hand_id = p1.loadURDF(args['hand_path'], useFixedBase=True,
                                basePosition=[0.0, 0.0, 0.05], flags=p1.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
            obj_id = p1.loadURDF(args['object_path'], basePosition=[0.0, 0.10, .05], flags=p1.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
            
            # Create TwoFingerGripper Object and set the initial joint positions
            hand = TwoFingerGripper(hand_id, path=args['hand_path'])

            obj = ObjectWithVelocity(obj_id, path=args['object_path'],name='obj_2')

            eval_goal_poses = GoalHolder(eval_pose_list,eval_names)

            goal_poses = RandomGoalHolder([0.02,0.065])

            state = StateRL(objects=[hand, obj, goal_poses], prev_len=args['pv'],eval_goals = eval_goal_poses)

            env = rl_env.ExpertEnv(hand=hand, obj=obj, hand_type=args['hand'], rand_start=args['rstart'])

            p1.configureDebugVisualizer(p1.COV_ENABLE_RENDERING,1)
            x = [0.03, 0, -0.03, -0.04, -0.03, 0, 0.03, 0.04]
            y = [-0.03, -0.04, -0.03, 0, 0.03, 0.04, 0.03, 0]
            action = rl_action.ExpertAction()
            reward = rl_reward.ExpertReward()
            manipulation = manipulation_phase_rl.ManipulationRL(
        hand, obj, x, y, state, action, reward, replay_buffer=None, args=args)
            manipulation.reset()
            manipulation.setup()
            start_state = state.get_state()
            childPipe.send([start_state['two_finger_gripper']['joint_angles']])
            # print('GOT TO THE END OF THE RESET THING')
            continue
        if message == _EXPLORE:
            a2 = np.array([-0.1,0.1,0.1,-0.1])
            # a3 = np.
            for i in range(150):
                manipulation.gym_pre_step(a2)
                manipulation.execute_action(pybullet_thing=p1)
                p1.stepSimulation()
                # print('just env stepped',rank, i)
                manipulation.exit_condition()
                manipulation.post_step()
            end_state = state.get_state()
            manipulation.reset()
            manipulation.setup()
            childPipe.send([end_state['two_finger_gripper']['joint_angles']])


            curr_joint_states = p1.getJointStates(hand_id, [0,1,3,4])
            a = [c[0] for c in curr_joint_states]
            print(a)
            continue
        if message == _CLOSE:
            childPipe.send(["close ok"])
            break
    childPipe.close()
  


if __name__ == "__main__":
    start = time.time()
    this_path = os.path.abspath(__file__)
    overall_path = os.path.dirname(os.path.dirname(os.path.dirname(this_path)))
    filepath = overall_path+'/demos/rl_demo/data/multiprocessing_spot/experiment_config.json'
    with open(filepath, 'r') as argfile:
        args = json.load(argfile)
    if args['action'] == 'Joint Velocity':
        args['ik_flag'] = False
    else:
        args['ik_flag'] = True
    mp.freeze_support()
    num_processes = 16
    processes = []
    nums=[0]*num_processes

    childPipes = []
    parentPipes = []
    xeval = [0.045, 0, -0.045, -0.06, -0.045, 0, 0.045, 0.06]
    yeval = [-0.045, -0.06, -0.045, 0, 0.045, 0.06, 0.045, 0]
    eval_names = ['SE','S','SW','W','NW','N','NE','E'] 
    eval_pose_list = [[i,j] for i,j in zip(xeval,yeval)]
    for pr in range(num_processes):
        parentPipe, childPipe = Pipe()
        parentPipes.append(parentPipe)
        childPipes.append(childPipe)

    for rank in range(num_processes):
        p = mp.Process(target=ExploreWorker, args=(rank, num_processes, childPipes[rank],  args, eval_pose_list))
        p.start()
        processes.append(p)


    for parentPipe in parentPipes:
        parentPipe.send([_RESET, "blaat"])

    positive_rewards = [0]*num_processes
    for k in range(num_processes):
        print("Start state",parentPipes[k].recv()[0])

    mid = time.time()
    for parentPipe in parentPipes:
        parentPipe.send([_EXPLORE, "blaat"])

    positive_rewards = [0]*num_processes
    for k in range(num_processes):
        positive_rewards[k] = parentPipes[k].recv()[0]
        # print("End state",positive_rewards[k])


    for parentPipe in parentPipes:
        parentPipe.send([_CLOSE, "pay2"])

    for p in processes:
        p.join()

    end = time.time()
    print('finished all the runs')
    print(f'total time required: {end-start}')
    print(f'just running time: {mid-start}')
