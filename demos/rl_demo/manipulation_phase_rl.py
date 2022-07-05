import pybullet as p
from mojograsp.simcore.phase import Phase
from mojograsp.simobjects.two_finger_gripper import TwoFingerGripper
from mojograsp.simobjects.object_base import ObjectBase
from mojograsp.simcore.state import State
from mojograsp.simcore.reward import Reward
from mojograsp.simcore.action import Action
import rl_controller
from mojograsp.simcore.replay_buffer import ReplayBufferDefault


from math import isclose


class ManipulationRL(Phase):

    def __init__(self, hand: TwoFingerGripper, cube: ObjectBase, x, y, state: State, action: Action, reward: Reward, replay_buffer: ReplayBufferDefault = None, args: dict = None):
        self.name = "manipulation"
        self.hand = hand
        self.cube = cube
        self.state = state
        self.action = action
        self.reward = reward
        self.terminal_step = 400
        self.timestep = 0
        self.episode = 0
        self.x = x
        self.y = y
        self.target = None
        self.goal_position = None
        # create controller
        self.controller = rl_controller.RLController(hand, cube, replay_buffer=replay_buffer, args=args)
        # self.controller = rl_controller.ExpertController(hand, cube)

    def setup(self):
        # reset timestep counter
        self.timestep = 0
        # rest contact loss counter in controller
        self.controller.num_contact_loss = 0
        # Get new goal position
        self.goal_position = [float(self.x[self.episode]), float(
            self.y[self.episode] + .16), 0]
        # set the new goal position for the controller
        self.controller.set_goal_position(self.goal_position)

    def pre_step(self):
        # Get the target action
        self.target = self.controller.get_next_action()
        # Set the next action before the sim is stepped for Action (Done so that replay buffer and record data work)
        self.action.set_action(self.target)
        # Set the current state before sim is stepped
        self.state.set_state()

    def execute_action(self):
        # Execute the target that we got from the controller in pre_step()
        p.setJointMotorControlArray(self.hand.id, jointIndices=self.hand.get_joint_numbers(),
                                    controlMode=p.POSITION_CONTROL, targetPositions=self.target)
        self.timestep += 1

    def post_step(self):
        # Set the reward from the given action after the step
        self.reward.set_reward(self.goal_position, self.cube)


    def exit_condition(self) -> bool:
        # If we reach 400 steps or the controller exit condition finishes we exit the phase
        if self.timestep > self.terminal_step or self.controller.exit_condition():
            self.controller.retry_count=0
            print(self.cube.get_curr_pose()[0][0],self.cube.get_curr_pose()[0][1]-0.16)
            return True
        return False

    def next_phase(self) -> str:
        # increment episode count and return next phase (None in this case)
        self.episode += 1
        return None

    def reset(self):
        self.episode = 0