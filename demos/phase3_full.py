import mojograsp
import pybullet as p
import math


class MoveHand(mojograsp.phase.Phase):
    def __init__(self, name=None):
        super().__init__()
        self.name = name
        self.terminal_step = 100
        state_path = '/Users/asar/Desktop/Grimm\'s Lab/Manipulation/PyBulletStuff/mojo-grasp/mojograsp/simcore/' \
                     'simmanager/State/simple_state.json'
        self.state = mojograsp.state_space.StateSpace(path=state_path)
        self.controller = mojograsp.controller_base.ControllerBase.create_instance(state_path=state_path,
                                                                                   controller_type='move')
        self.curr_action = None
        self.curr_action_profile = None
        self.Action = mojograsp.action_class.Action()
        self.reward = mojograsp.reward_class.Reward('/Users/asar/Desktop/Grimm\'s Lab/Manipulation/PyBulletStuff'
                                                    '/mojo-grasp/mojograsp/simcore/simmanager/Reward/reward_demo.json')

    def setup(self):
        roll_fric = 0.01
        p.changeDynamics(self._sim.objects.id, -1, mass=0.1, rollingFriction=roll_fric)
        p.changeDynamics(self._sim.hand.id, 1, rollingFriction=roll_fric)
        p.changeDynamics(self._sim.hand.id, 3, rollingFriction=roll_fric)
        print("{} setup".format(self.name))
        print("{} executing".format(self.name))
        self.controller.iterator = 0
        self.controller.data_over = False

    def execute_action(self, action):
        p.setJointMotorControlArray(self._sim.hand.id, jointIndices=self._sim.hand.actuation.get_joint_index_numbers(),
                                    controlMode=p.POSITION_CONTROL, targetPositions=action)

    def phase_exit_condition(self, curr_step):
        if curr_step >= self.terminal_step or self.controller.data_over:
            print(curr_step, self.controller.data_over)
            done = True
        else:
            done = False
        return done

    def phase_complete(self):
        print("completed")
        return


if __name__ == '__main__':
    pass
