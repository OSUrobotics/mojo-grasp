import mojograsp
import pybullet as p
import math
import pathlib


class CloseHand(mojograsp.phase.Phase):
    def __init__(self, name=None):
        super().__init__()
        self.target_pos = [0.7, -1.4, -0.7, 1.4]
        self.name = name
        self.terminal_step = 30
        current_path = str(pathlib.Path().resolve())
        state_path = current_path+"/state_action_reward/simple_state.json"
        self.state = mojograsp.state_space.StateSpace(path=state_path)
        self.controller = mojograsp.controller_base.ControllerBase.create_instance(state_path=state_path,
                                                                                   controller_type='close')
        self.curr_action = None
        self.curr_action_profile = None
        action_path = current_path+"/state_action_reward/action.json"
        self.Action = mojograsp.action_class.Action(json_path=action_path)
        self.reward = None # mojograsp.reward_class.Reward()

    def setup(self):
        roll_fric = 0.01
        # object
        p.changeDynamics(self._sim.objects.id, -1, mass=0.04, rollingFriction=roll_fric)
        # distal
        p.changeDynamics(self._sim.hand.id, 1, mass=0.03, rollingFriction=roll_fric)
        p.changeDynamics(self._sim.hand.id, 3, mass=0.03, rollingFriction=roll_fric)
        # proximal
        p.changeDynamics(self._sim.hand.id, 0, mass=0.02, rollingFriction=roll_fric)
        p.changeDynamics(self._sim.hand.id, 2, mass=0.02, rollingFriction=roll_fric)
        # print("{} setup".format(self.name))
        # self._sim.objects.set_curr_pose([0.00, 0.17, 0.0], self._sim.objects.start_pos[self._sim.objects.id][1])
        # print("{} executing".format(self.name))

    def execute_action(self, action):
        p.setJointMotorControlArray(self._sim.hand.id, jointIndices=self._sim.hand.actuation.get_joint_index_numbers(), controlMode=p.POSITION_CONTROL,
                                    targetPositions=action)

    def phase_exit_condition(self, curr_step):
        count = 0
        for curr_joint, given_joint in zip(self.state.get_value('StateGroup_Angle_JointState'), self.target_pos):
            if math.isclose(curr_joint, given_joint, abs_tol=1e-4):
                count += 1
        if count == len(self._sim.hand.actuation.get_joint_index_numbers()) or curr_step >= self.terminal_step:
            # print(curr_step)
            done = True
        else:
            done = False
        return done

    def phase_complete(self):
        # print("move rl")
        # return "move expert"
        return "move rl"
