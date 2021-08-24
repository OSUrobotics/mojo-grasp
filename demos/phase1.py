import mojograsp
import pybullet as p


class OpenHand(mojograsp.phase.Phase):
    def __init__(self, name=None):
        super().__init__()
        self.joint_nums = self._sim.hand.actuation.get_joint_index_numbers()
        self.target_pos = [1.57, 0, -1.57, 0]
        self.name = name
        self.terminal_step = 10
        state_path = '/Users/asar/Desktop/Grimm\'s Lab/Manipulation/PyBulletStuff/mojo-grasp/mojograsp/simcore/simmanager/State/simple_state.json'
        self.state = mojograsp.state_space.StateSpace(path=state_path)
        self.controller = mojograsp.controller_base.GenericController('open', state_path=state_path)
        self.curr_action = None
        self.curr_action_profile = None
        self.Action = mojograsp.action_class.Action()
        self.reward = None # mojograsp.reward_class.Reward()

    def setup(self):
        print("{} setup".format(self.name))
        self.joint_nums = self._sim.hand.actuation.get_joint_index_numbers()
        print("{} executing".format(self.name))

    def execute_action(self, action):
        p.setJointMotorControlArray(self._sim.hand.id, jointIndices=self.joint_nums, controlMode=p.POSITION_CONTROL,
                                    targetPositions=action)

    def phase_exit_condition(self, curr_step):
        # state = {'joint_angles': self.get_joint_angles()}
        if curr_step >= self.terminal_step:
            print("Phase: {} completed in {} steps".format(self.name, curr_step))
            return True
        return False

    def phase_complete(self):
        print("close phase")
        return "close phase"
