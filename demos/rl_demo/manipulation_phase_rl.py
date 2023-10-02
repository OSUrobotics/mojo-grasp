import pybullet as p
from mojograsp.simcore.phase import Phase
from mojograsp.simobjects.two_finger_gripper import TwoFingerGripper
from mojograsp.simobjects.object_base import ObjectBase
from mojograsp.simcore.state import State
from mojograsp.simcore.reward import Reward
from mojograsp.simcore.action import Action
from demos.rl_demo import rl_controller
from mojograsp.simcore.replay_buffer import ReplayBufferDefault
from numpy.random import shuffle
from math import isclose



class ManipulationRL(Phase):

    def __init__(self, hand: TwoFingerGripper, cube: ObjectBase, x, y, state: State, action: Action, reward: Reward, env, replay_buffer: ReplayBufferDefault = None, args: dict = None,physicsClientId = None):
        self.name = "manipulation"
        self.hand = hand
        self.cube = cube
        self.state = state
        self.action = action
        self.reward = reward
        self.eval = False
        self.env = env
        try:
            self.terminal_step = args['tsteps']
            self.eval_terminal_step = args['eval-tsteps']
        except KeyError:
            self.terminal_step = 150
        self.timestep = 0
        self.episode = 0
        self.x = x
        self.y = y
        self.use_ik = args['ik_flag']
        print('ARE WE USING IK', self.use_ik)
        
        self.target = None
        self.goal_position = None
        # create controller
        self.interp_ratio = int(240/args['freq'])
        if args['model'] == 'gym':
            self.controller = rl_controller.GymController(hand, cube, args=args)
        else:
            self.controller = rl_controller.RLController(hand, cube, replay_buffer=replay_buffer, args=args)
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

        # self.goal_position = [float(self.x[self.episode]), float(
        #     self.y[self.episode] + .10), 0]
        
        temp_position = self.state.objects[-1].get_data()
        self.goal_position = [temp_position['goal_pose'][0], temp_position['goal_pose'][1]+0.1, 0]
        # self.goal_position[1] += 0.1
        # print(self.goal_position)
        # set the new goal position for the controller
        self.controller.set_goal_position(self.goal_position)
        self.state.init_state()
        start_state = self.state.get_state()
        self.reward.setup_reward(start_state['obj_2']['pose'])

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
        if self.use_ik:
            self.target, self.actor_portion = self.controller.find_angles(gym_action)
            # print('manipulation phase rl',self.actor_portion)
        else:
            self.target, self.actor_portion =  self.controller.find_angles(gym_action)

        # Set the next action before the sim is stepped for Action (Done so that replay buffer and record data work)
        self.action.set_action(self.target, self.actor_portion)
        # Set the current state before sim is stepped
        # self.state.set_state()

    def execute_action(self, action_to_execute=None, pybullet_thing = None):
        # Execute the target that we got from the controller in pre_step()

        for i in range(self.interp_ratio):
            if pybullet_thing is not None:
                # temp = pybullet_thing.getJointStates(self.hand.id, [0,1,3,4])
                goal_angs = self.action.get_joint_angles()
                # print(f'goal angles {goal_angs}')
                # print('joint states before motion', temp[0][0], temp[1][0], temp[2][0], temp[3][0])
                # print('joint goals',goal_angs)
                pybullet_thing.setJointMotorControlArray(self.hand.id, jointIndices=self.hand.get_joint_numbers(),
                                            controlMode=p.POSITION_CONTROL, targetPositions=goal_angs, positionGains=[0.8,0.8,0.8,0.8], forces=[0.4,0.4,0.4,0.4])
            elif action_to_execute:
                # temp = p.getJointStates(self.hand.id, [0,1,3,4])
                # print('joint states before motion', temp[0][0], temp[1][0], temp[2][0], temp[3][0])
                # print('joint goals', action_to_execute)
                p.setJointMotorControlArray(self.hand.id, jointIndices=self.hand.get_joint_numbers(),
                                            controlMode=p.POSITION_CONTROL, targetPositions=action_to_execute, positionGains=[0.8,0.8,0.8,0.8], forces=[0.4,0.4,0.4,0.4])
            else:
                goal_angs = self.action.get_joint_angles()
                # print(f'goal angles {goal_angs}')
                # temp = p.getJointStates(self.hand.id, [0,1,3,4])
                # print('joint states before motion', temp[0][0], temp[1][0], temp[2][0], temp[3][0])
                # print('joint goals',goal_angs)
                # print('no action given',self.action.get_joint_angles())
                p.setJointMotorControlArray(self.hand.id, jointIndices=self.hand.get_joint_numbers(),
                                            controlMode=p.POSITION_CONTROL, targetPositions=goal_angs, positionGains=[0.8,0.8,0.8,0.8], forces=[0.4,0.4,0.4,0.4])
            self.env.step()
            # if pybullet_thing is not None:
            #     temp = pybullet_thing.getJointStates(self.hand.id, [0,1,3,4])
            # elif action_to_execute:
            #     temp = p.getJointStates(self.hand.id, [0,1,3,4])
            # else:
            #     temp = p.getJointStates(self.hand.id, [0,1,3,4])

            # print('joint states after motion', temp[0][0], temp[1][0], temp[2][0], temp[3][0])
            
        self.timestep += 1

    def post_step(self):
        # Set the reward from the given action after the step
        self.reward.set_reward(self.goal_position, self.cube, self.hand, self.controller.final_reward)

    def get_episode_info(self):
        # DO NOT USE UNLESS THIS IS GYM WRAPPER
        self.state.set_state()
        return self.state.get_state(), self.reward.get_reward()

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

    def reset(self):
        print('still reseting')
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