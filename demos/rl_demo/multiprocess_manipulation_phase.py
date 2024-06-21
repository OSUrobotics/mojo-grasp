from mojograsp.simcore.phase import Phase
from mojograsp.simobjects.two_finger_gripper import TwoFingerGripper
from mojograsp.simobjects.object_base import ObjectBase
from mojograsp.simcore.state import State
from mojograsp.simcore.reward import Reward
from mojograsp.simcore.action import Action
from demos.rl_demo import multiprocess_control
from mojograsp.simcore.replay_buffer import ReplayBufferDefault
from numpy.random import shuffle
from math import isclose
from PIL import Image

# import numpy as np
# import matplotlib.pyplot as plt

class MultiprocessManipulation(Phase):

    def __init__(self, hand: TwoFingerGripper, cube: ObjectBase, state: State, action: Action, reward: Reward, env, replay_buffer: ReplayBufferDefault = None, args: dict = None,physicsClientId = None, hand_type=None):
        self.name = "manipulation"
        self.hand = hand
        self.cube = cube
        self.state = state
        self.action = action
        self.reward = reward
        self.eval = False
        self.env = env
        self.p = self.env.p
        try:
            self.terminal_step = args['tsteps']
            self.eval_terminal_step = args['eval-tsteps']
        except KeyError:
            self.terminal_step = 150
        self.timestep = 0
        self.episode = 0
        self.use_ik = args['ik_flag']
        print('ARE WE USING IK', self.use_ik)
        self.image_path = args['save_path'] + 'Videos/'
        self.target = None
        # create controller
        self.interp_ratio = int(240/args['freq'])
        self.controller = multiprocess_control.MultiprocessController(self.p, hand, cube, args=args,hand_type=hand_type)
        self.end_val = 0
        self.camera_view_matrix = self.p.computeViewMatrix((0.0,0.1,0.5),(0.0,0.1,0.005), (0.0,1,0.0))
        self.camera_projection_matrix = self.p.computeProjectionMatrixFOV(60,4/3,0.1,0.9)
        # self.controller = rl_controller.ExpertController(hand, cube)

    def setup(self):
        # print('episode according to maniprl', self.episode)
        # reset timestep counter
        self.timestep = 0
        # rest contact loss counter in controller
        self.controller.num_contact_loss = 0
        # Get new goal position
        # self.goal_position = [float(self.x[self.episode]), float(
        #     self.y[self.episode] + .1067), 0]

        # self.goal_position = [float(self.x[self.episode]), float(
        #     self.y[self.episode] + .10), 0]
        
        temp_position = self.state.objects[-1].get_data()
        # self.goal_position[1] += 0.1
        # print(self.goal_position)
        # set the new goal position for the controller
        self.controller.pre_step()
        self.state.init_state()
        start_state = self.state.get_state()
        # print('start state',start_state['f1_pos'],start_state['f2_pos'])
        self.reward.setup_reward(start_state)

    def pre_step(self):
        # Get the target action
        S = self.state.get_state()
        if self.use_ik:
            self.target, self.actor_portion = self.controller.get_next_IK_action(S)
            # print('manipulation phase rl',self.actor_portion)
        else:
            self.target, self.actor_portion = self.controller.get_next_action(S)

        # Set the next action before the sim is stepped for Action (Done so that replay buffer and record data work)
        self.action.set_action(self.target, self.actor_portion)
        # Set the current state before sim is stepped
        # self.state.set_state()

    def gym_pre_step(self, gym_action):
        # Get the target action
        # print('prestepping by going into find angles in controller')
        # print(gym_action)
        self.target, self.actor_portion =  self.controller.find_angles(gym_action)

        # Set the next action before the sim is stepped for Action (Done so that replay buffer and record data work)
        self.action.set_action(self.target, self.actor_portion)
        # Set the current state before sim is stepped
        # self.state.set_state()

    def execute_action(self, action_to_execute=None, pybullet_thing = None, viz = False):
        # Execute the target that we got from the controller in pre_step()
        errs = []
        goals =[ ]
        actual = []
        forces = []
        f2 = []
        f3 = []
        f4 = []
        for i in range(self.interp_ratio):
            # print(self.timestep,i)
            if pybullet_thing is not None:
                # temp = pybullet_thing.getJointStates(self.hand.id, [0,1,3,4])
                goal_angs = self.action.get_joint_angles()
                # print(f'goal angles {goal_angs}')
                # print('joint states before motion', temp[0][0], temp[1][0], temp[2][0], temp[3][0])
                # print('joint goals',goal_angs)
                pybullet_thing.setJointMotorControlArray(self.hand.id, jointIndices=self.hand.get_joint_numbers(),
                                            controlMode=self.p.POSITION_CONTROL, targetPositions=goal_angs, positionGains=[0.8,0.8,0.8,0.8], forces=[0.4,0.4,0.4,0.4])
            elif action_to_execute:
                # temp = p.getJointStates(self.hand.id, [0,1,3,4])
                # print('joint states before motion', temp[0][0], temp[1][0], temp[2][0], temp[3][0])
                # print('joint goals', action_to_execute
                self.p.setJointMotorControlArray(self.hand.id, jointIndices=self.hand.get_joint_numbers(),
                                            controlMode=self.p.POSITION_CONTROL, targetPositions=action_to_execute, positionGains=[0.8,0.8,0.8,0.8], forces=[0.4,0.4,0.4,0.4],
                                            velocityGains=[0,0,0,0])
            else:
                goal_angs = self.action.get_joint_angles()
                # print(f'goal angles {goal_angs}')
                
                temp = self.p.getJointStates(self.hand.id, [0,1,3,4])
                # print('joint states before motion', temp[0][0], temp[1][0], temp[2][0], temp[3][0])
                # print('joint goals',goal_angs)
                # print('errors', temp[0][0]-goal_angs[0], temp[1][0]-goal_angs[1], temp[2][0]-goal_angs[2], temp[3][0]-goal_angs[3])
                # errs.append([temp[0][0]-goal_angs[0], temp[1][0]-goal_angs[1], temp[2][0]-goal_angs[2], temp[3][0]-goal_angs[3]])
                # goals.append(goal_angs[0])
                # actual.append(temp[0][0])
                # print(self.hand.get_joint_numbers())
                # print('no action given',self.action.get_joint_angles())
                self.p.setJointMotorControlArray(self.hand.id, jointIndices=self.hand.get_joint_numbers(),
                                            controlMode=self.p.POSITION_CONTROL, targetPositions=goal_angs, positionGains=[0.8,0.8,0.8,0.8], forces=[0.4,0.4,0.4,0.4])

            self.env.step()
            
            if viz and (i%10 == 0):
                img = self.p.getCameraImage(640, 480,viewMatrix=self.camera_view_matrix,
                                        projectionMatrix=self.camera_projection_matrix,
                                        shadow=1,
                                        lightDirection=[1, 1, 1])
                img = Image.fromarray(img[2])
                temp = 'eval'
                img.save(self.image_path+ temp + '_frame_'+ str(self.timestep)+'_'+str(i)+'.png')

            # temp = self.p.getJointStates(self.hand.id, [0,1,3,4])
            # print('joint states after motion', temp[0][0], temp[1][0], temp[2][0], temp[3][0])
            # print('forces ', temp[0][3], temp[1][3], temp[2][3], temp[3][3])
            # forces.append(temp[0][3])
            # f2.append(temp[1][3])
            # f3.append(temp[2][3])
            # f4.append(temp[3][3])
        # errs = np.array(errs)
        # plt.plot(range(len(errs)), errs[:,0])
        # fig, (ax1, ax2) = plt.subplots(2, 1)
        # ax1.plot(range(len(goals)),goals)
        # ax1.plot(range(len(actual)),actual)
        # plt.legend(['goal positions','actual positions'])
        # plt.plot(range(len(errs)), errs[:,1])
        # plt.plot(range(len(errs)), errs[:,2])
        # plt.plot(range(len(errs)), errs[:,3])
        # ax1.set_xlabel('sim-step (1/240 s)')
        # ax1.set_ylabel('Error (rad)')
        # ax1.set_grid(True)

        # ax2.plot(range(len(forces)),forces)
        # ax2.plot(range(len(f2)),f2)
        # ax2.plot(range(len(f3)),f3)
        # ax2.plot(range(len(f4)),f4)

        # plt.show()
        self.timestep += 1

    def post_step(self):
        # Set the reward from the given action after the step
        self.reward.set_reward(self.state.get_goal(), self.cube, self.hand, self.controller.final_reward)

    def next_goal(self):
        new_goal = self.state.next_run()
        # print('new goal from manip_phase.new_goal', new_goal)
        self.reward.update_start(new_goal, self.cube)

    def get_episode_info(self):
        # DO NOT USE UNLESS THIS IS GYM WRAPPER
        self.state.set_state()
        return self.state.get_state(), self.reward.get_reward()

    def get_built_sub_state(self, statekeys):
        print('not implemented, use the build state in subpolicy holder or fix this')

    def get_state(self):
        return self.state.get_state()

    def set_goal(self, goal_list):
        self.state.set_goal(goal_list)

    def exit_condition(self, eval_exit=False) -> bool:
        # If we reach 400 steps or the controller exit condition finishes we exit the phase
        if eval_exit:
            if self.timestep > self.eval_terminal_step:
                self.controller.retry_count=0
                # print('exiting in manipulation phase rl', self.timestep, self.terminal_step)
                return True
        else:
            if self.timestep > self.terminal_step:# or self.controller.exit_condition(self.terminal_step - self.timestep):
                self.controller.retry_count=0
                # print('exiting in manipulation phase rl', self.timestep, self.terminal_step)
                return True
        return False

    def next_phase(self) -> str:
        # increment episode count and return next phase (None in this case)
        if not self.eval:
            self.episode += 1
        self.state.next_run()
        return None

    def next_ep(self) -> str:
        # increment episode count and return next goal
        if not self.eval:
            self.episode += 1
        _, fingerys = self.state.next_run()
        return self.state.objects[-1].get_data(), fingerys

    def reset(self):
        # temp = list(range(len(self.x)))
        
        # shuffle(temp)
        # self.x = [self.x[i] for i in temp]
        # self.y = [self.y[i] for i in temp]
        
        self.episode = 0
        
        self.state.reset()

    def load_policy(self, filename):
        self.controller.load_policy(filename)
        
    def evaluate(self):
        self.controller.evaluate()
        self.eval = True
        
    def train(self):
        self.controller.train()
        self.eval = False