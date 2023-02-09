import pybullet as p
from mojograsp.simcore.phase import Phase
from mojograsp.simobjects.two_finger_gripper import TwoFingerGripper
from mojograsp.simobjects.object_base import ObjectBase
from mojograsp.simcore.state import State
from mojograsp.simcore.reward import Reward
from mojograsp.simcore.action import Action
import rl_controller
from mojograsp.simcore.replay_buffer import ReplayBufferDefault
from numpy.random import shuffle

from math import isclose


class ManipulationRL(Phase):

    def __init__(self, hand: TwoFingerGripper, cube: ObjectBase, x, y, state: State, action: Action, reward: Reward, replay_buffer: ReplayBufferDefault = None, args: dict = None, tbname = None):
        self.name = "manipulation"
        self.hand = hand
        self.cube = cube
        self.state = state
        self.action = action
        self.reward = reward
        self.terminal_step = 150
        self.timestep = 0
        self.episode = 0
        self.x = x
        self.y = y
        try:
            self.use_ik = args['ik_flag']
        except:
            self.use_ik = False
        # print('x and y', x,y)
        
        self.target = None
        self.goal_position = None
        # create controller
        self.controller = rl_controller.RLController(hand, cube, replay_buffer=replay_buffer, args=args, tbname=tbname)
        self.end_val = 0
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

        self.goal_position = [float(self.x[self.episode]), float(
            self.y[self.episode] + .16), 0]
        # print(self.goal_position)
        # set the new goal position for the controller
        self.controller.set_goal_position(self.goal_position)

    def pre_step(self):
        # Get the target action
        if self.use_ik:
            self.target, self.actor_portion = self.controller.get_next_IK_action()
        else:
            self.target, self.actor_portion = self.controller.get_next_action()

        # Set the next action before the sim is stepped for Action (Done so that replay buffer and record data work)
        self.action.set_action(self.target, self.actor_portion)
        # Set the current state before sim is stepped
        self.state.set_state()

    def execute_action(self, action_to_execute=None):
        # Execute the target that we got from the controller in pre_step()
        if action_to_execute:
            p.setJointMotorControlArray(self.hand.id, jointIndices=self.hand.get_joint_numbers(),
                                        controlMode=p.POSITION_CONTROL, targetPositions=action_to_execute)
        else:
            p.setJointMotorControlArray(self.hand.id, jointIndices=self.hand.get_joint_numbers(),
                                        controlMode=p.POSITION_CONTROL, targetPositions=self.target)
        self.timestep += 1

    def post_step(self):
        # Set the reward from the given action after the step
        self.reward.set_reward(self.goal_position, self.cube, self.hand, self.controller.final_reward)


    def exit_condition(self) -> bool:
        # If we reach 400 steps or the controller exit condition finishes we exit the phase
        if self.timestep > self.terminal_step:# or self.controller.exit_condition(self.terminal_step - self.timestep):
            self.controller.retry_count=0
            print('exitiny in manipulation phase rl', self.timestep, self.terminal_step)
            return True
        return False

    def next_phase(self) -> str:
        # increment episode count and return next phase (None in this case)
        self.episode += 1
        self.state.next_run()
        return None

    def reset(self):
        print('still reseting')
        # temp = list(range(len(self.x)))
        
        # shuffle(temp)
        # self.x = [self.x[i] for i in temp]
        # self.y = [self.y[i] for i in temp]
        
        self.episode = 0
        
        self.state.reset()
