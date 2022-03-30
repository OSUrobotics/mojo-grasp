from mojograsp.simcore.episode import Episode
import pybullet as p


class EpisodeConfig(Episode):
    def setup(self):
        pass

    def reset(self):
        hand = self.env.hand
        object = self.env.object
        for i in hand.joint_dict.values():
            p.resetJointState(hand.id, i, 0)
        p.resetBasePositionAndOrientation(
            object.id, [0.0, 0.17, .06], [0, 0, 0, 1])

    def post_episode(self):
        pass
