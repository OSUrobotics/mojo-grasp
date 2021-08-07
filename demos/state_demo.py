import mojograsp
import pybullet as p
import numpy as np


class Controller:
    def __init__(self):
        pass

    def select_action(self):
        """
        This controller is defined to open the hand.
        Thus the action will always be a constant action
        of joint position to be reached
        :return: action: action to be taken in accordance with action space
        """
        action = [1.57, 0, -1.57, 0]
        return action


class OpenHand(mojograsp.phase.Phase):
    def __init__(self, name=None):
        super().__init__()
        self.joint_nums = self._sim.hand.actuation.get_joint_index_numbers()
        self.target_pos = [1.57, 0, -1.57, 0]
        self.controller = Controller()
        self.name = name
        self.terminal_step = 50
        state_path = '/Users/asar/Desktop/Grimm\'s Lab/Manipulation/PyBulletStuff/mojo-grasp/mojograsp/simcore/simmanager/State/simple_state.json'
        self.state = mojograsp.state_space.StateSpace(path=state_path)
        self.curr_action=None

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
        print("Done")
        return


if __name__ == '__main__':
    # setting up simmanager/physics server
    manager = mojograsp.simmanager.SimManager(rl=False)
    # setting camera
    p.resetDebugVisualizerCamera(cameraDistance=.4, cameraYaw=0, cameraPitch=-45, cameraTargetPosition=[.1, 0, .1])
    hand_path = '/Users/asar/Desktop/Grimm\'s Lab/Manipulation/PyBulletStuff/mojo-grasp/hand_generation/hand_models/2v2_nosensors/2v2_nosensors.urdf'
    # hand_path = "/Users/asar/PycharmProjects/InHand-Manipulation/ExampleSimWorld-Josh/2v2_hands/999/testing.sdf"

    object_path = '/Users/asar/Desktop/Grimm\'s Lab/Manipulation/PyBulletStuff/mojo-grasp/hand_generation/object_models/2v2_nosensors/2v2_nosensors_cuboid_small.urdf'

    hand = mojograsp.hand.Hand(hand_path, fixed=True)
    cube = mojograsp.objectbase.ObjectBase(object_path, fixed=False)

    sim_env = mojograsp.environment.Environment(hand=hand, objects=[cube], steps=1)

    mojograsp.phase.Phase._sim = sim_env
    open = OpenHand('open phase')
    mojograsp.state_metric_base.StateMetricBase._sim = sim_env


    # print("STATE DATA: {}".format(state.data['Angle_JointState'].get_xml_geom_name('F1')))

    # manager.add_state_space(state)

    manager.add_phase(open.name, open, start=True)
    print("STATE: {}".format(open.state.update()))

    # running simulation
    manager.run()
    # stalling so it doesnt exit
    manager.stall()