import mojograsp
import pybullet as p
import math
import pathlib
import pandas as pd


class MoveRL(mojograsp.phase.Phase):
    def __init__(self, name=None):
        super().__init__()
        self.name = name
        self.terminal_step = 30
        current_path = str(pathlib.Path().resolve())
        state_path = current_path+"/state_action_reward/simple_state.json"
        self.state = mojograsp.state_space.StateSpace(path=state_path)
        self.controller = mojograsp.controller_base.ControllerBase.create_instance(state_path=state_path,
                                                                                   controller_type='rl')
        self.curr_action = None
        self.curr_action_profile = None
        self.Action = mojograsp.action_class.Action()
        self.reward = mojograsp.reward_class.Reward(current_path + '/state_action_reward/reward_demo.json')
        self.save_data_dict = {}

    def setup(self):
        self.save_data_dict = {}
        roll_fric = 0.01
        p.changeDynamics(self._sim.objects.id, -1, mass=0.1, rollingFriction=roll_fric)
        p.changeDynamics(self._sim.hand.id, 1, rollingFriction=roll_fric)
        p.changeDynamics(self._sim.hand.id, 3, rollingFriction=roll_fric)
        # print("{} setup".format(self.name))
        # print("{} executing".format(self.name))
        self.controller.iterator = 0
        self.controller.data_over = False
        self.controller.dir = mojograsp.phase.Phase._sim.curr_dir

    def execute_action(self, action):
        p.setJointMotorControlArray(self._sim.hand.id, jointIndices=self._sim.hand.actuation.get_joint_index_numbers(),
                                    controlMode=p.POSITION_CONTROL, targetPositions=action)

    def phase_exit_condition(self, curr_step):
        if curr_step >= self.terminal_step:
            # print(curr_step)
            done = True
        else:
            done = False
        return done

    def phase_complete(self):
        self.save_file()
        # print("completed")
        return

    def get_cube_in_start_pos(self):
        cube_pos, cube_orn = self._sim.get_obj_curr_pose()
        init_cube_pose = self._sim.objects.start_pos[self._sim.objects.id]
        print(cube_pos, cube_orn, init_cube_pose[0], init_cube_pose[1])
        new_pose = p.multiplyTransforms(init_cube_pose[0], init_cube_pose[1], cube_pos, cube_orn)
        new_pos = [new_pose[0][0], new_pose[0][1], new_pose[0][2]]
        self.add_column(column_name='Cube_pos_in_start_pos', column_data=new_pos)

    def post_step(self):
        self.add_column(column_name='Phase', column_data='Move')
        self.add_column(column_name='human_cube_pos', column_data=None)
        self.add_column(column_name='human_cube_orn', column_data=None)
        cube_pos, cube_orn = self._sim.get_obj_curr_pose()
        self.add_column(column_name='Cube_Pos', column_data=cube_pos)
        self.add_column(column_name='Cube_Orn', column_data=cube_orn)
        self.get_cube_in_start_pos()

    def add_column(self, column_name='', column_data=None):
        if column_name not in self.save_data_dict.keys():
            self.save_data_dict.update({column_name: [column_data]})
        else:
            self.save_data_dict[column_name].append(column_data)

    def save_file(self):
        print("DICT: {}".format(self.save_data_dict))
        df = pd.DataFrame.from_dict(self.save_data_dict)
        print("DF {}".format(df.items))
        df.to_csv('2v2_RL_agent_with_expected_20k_train_{}'.format(self._sim.curr_dir)+'_save_data.csv')


if __name__ == '__main__':
    pass
