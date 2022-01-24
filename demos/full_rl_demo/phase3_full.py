import mojograsp
import pybullet as p
import math
import pathlib


class MoveHand(mojograsp.phase.Phase):
    def __init__(self, name=None):
        super().__init__()
        self.name = name
        self.terminal_step = 100
        current_path = str(pathlib.Path().resolve())
        state_path = current_path+"/state_action_reward/simple_state.json"
        self.state = mojograsp.state_space.StateSpace(path=state_path)
        self.controller = mojograsp.controller_base.ControllerBase.create_instance(state_path=state_path,
                                                                                   controller_type='move')
        self.curr_action = None
        self.curr_action_profile = None
        action_path = current_path+"/state_action_reward/action.json"
        self.Action = mojograsp.action_class.Action(json_path=action_path)
        
        self.reward = mojograsp.reward_class.Reward(current_path+'/state_action_reward/reward_demo.json')

    def setup(self):
        # print("{} setup".format(self.name))
        # print("{} executing".format(self.name))
        self.controller.dir = mojograsp.phase.Phase._sim.curr_dir
        self.controller.sub = mojograsp.phase.Phase._sim.curr_sub
        self.controller.trial = mojograsp.phase.Phase._sim.curr_trial
        self.controller.trial_type = mojograsp.phase.Phase._sim.curr_trial_type
        if self.controller.trial_type == 'human':
            self.controller.filename = "/Users/asar/PycharmProjects/InHand-Manipulation/Human Study Data/" \
                            "asterisk_test_data_for_anjali/trial_paths/not_normalized/{}_2v2_{}_n_{}.csv".format(self.controller.sub, self.controller.dir, self.controller.trial)
        elif self.controller.trial_type == 'expected':
            self.controller.filename = "/Users/asar/PycharmProjects/InHand-Manipulation/Human Study Data/" \
                                       "expected_data/exp_2v2_{}_n_1.csv".format(self.controller.dir)
        else:
            print("Wrong controller trial type!!")
            raise ValueError
        mojograsp.phase.Phase._sim.set_obj_target_pose(self.controller.dir)
        self.controller.object_poses_expert = self.controller.extract_data_from_file()
        self.controller.iterator = 0
        self.controller.data_len = len(self.controller.object_poses_expert)
        self.controller.data_over = False

    def execute_action(self, action):
        p.setJointMotorControlArray(self._sim.hand.id, jointIndices=self._sim.hand.actuation.get_joint_index_numbers(),
                                    controlMode=p.POSITION_CONTROL, targetPositions=action)

    def phase_exit_condition(self, curr_step):
        if curr_step >= self.terminal_step or self.controller.data_over:
            done = True
        else:
            done = False
        return done

    def phase_complete(self):
        # print("completed")
        return


if __name__ == '__main__':
    pass
