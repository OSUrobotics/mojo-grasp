import unittest
from mojograsp.simcore.DDPGfD import DDPGfD
from mojograsp.simcore.replay_buffer import ReplayBufferDefault

class DDPGfDBaseTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = DDPGfD()

class RollRewardTest(DDPGfDBaseTest):
    def runTest(self):
        reward_batch = [[10, 5, 2],[1.0,1.0]]
        sum_rewards, num_rewards = self.policy.calc_roll_rewards(reward_batch)
        self.assertEqual(sum_rewards[0][0],16.95505,
                         'incorrectly counting reward')
        self.assertEqual(sum_rewards[1][0],1.995,
                         'incorrectly counting reward')
        self.assertEqual(num_rewards[0][0],3,
                         'incorrectly counting number of rewards')
        self.assertEqual(num_rewards[1][0],2,
                         'incorrectly counting number of rewards')

# class CollectBatchTest(DDPGfDBaseTest):
#     def runTest(self):

if __name__ == '__main__':
    unittest.main()