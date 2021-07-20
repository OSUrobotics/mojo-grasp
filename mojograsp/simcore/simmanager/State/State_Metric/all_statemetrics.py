import pybullet as p
from mojograsp.simcore.simmanager.State.State_Metric.state_metric import StateMetricPyBullet


class Angle(StateMetricPyBullet):

    def update(self, keys):  # this function finds either the joint angles or the x and z angle,
        """
        Get the current joint angles
        Stores in self.curr_joint_angle : current joint angles as a list
        """
        curr_joint_poses = []
        curr_joint_states = p.getJointStates(self.hand.id, jointIndices=range(0, self.hand.num_joints))
        for joint in range(0, len(curr_joint_states)):
            curr_joint_poses.append(curr_joint_states[joint][0])
        self.data.set_value( curr_joint_poses)
        return self.data.value


class Position(StateMetricPyBullet):

    def update(self, keys):
        pass