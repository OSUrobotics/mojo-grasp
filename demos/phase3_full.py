import mojograsp
import pybullet as p
import math


class MoveHand(mojograsp.phase.Phase):
    def __init__(self, name=None):
        super().__init__()
        self.target_pos = [0.6, -0.9, -0.6, 0.9]
        self.name = name
        self.terminal_step = 10
        state_path = '/Users/asar/Desktop/Grimm\'s Lab/Manipulation/PyBulletStuff/mojo-grasp/mojograsp/simcore/' \
                     'simmanager/State/simple_state.json'
        self.state = mojograsp.state_space.StateSpace(path=state_path)
        self.controller = mojograsp.controller_base.ControllerBase.create_instance(state_path=state_path,
                                                                                   controller_type='close')
        self.curr_action = None
        self.curr_action_profile = None
        self.Action = mojograsp.action_class.Action()
        self.reward = None # mojograsp.reward_class.Reward()

    def setup(self):
        print("{} setup".format(self.name))
        self.joint_nums = self._sim.hand.actuation.get_joint_index_numbers()
        print("{} executing".format(self.name))

    def execute_action(self, action):
        p.setJointMotorControlArray(self._sim.hand.id, jointIndices=self._sim.hand.actuation.get_joint_index_numbers(),
                                    controlMode=p.POSITION_CONTROL, targetPositions=action)

    def phase_exit_condition(self, curr_step):
        count = 0
        for curr_joint, given_joint in zip(self.state.get_value('Angle_JointState'), self.target_pos):
            if math.isclose(curr_joint, given_joint, abs_tol=1e-4):
                count += 1
        if count == len(self._sim.hand.actuation.get_joint_index_numbers()) or curr_step >= self.terminal_step:
            print(curr_step)
            done = True
        else:
            done = False
        return done

    def phase_complete(self):
        print("completed")
        return


if __name__ == '__main__':
    pass
