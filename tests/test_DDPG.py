# project/test.py

import unittest
from mojograsp.simcore.DDPGfD import DDPGfD_priority
import pickle as pkl
import numpy as np
import torch

def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            return False
    return True
        

class TestDDPG(unittest.TestCase):

    def setUp(self):
        with open('./test_configs/experiment_config.pkl','rb') as configfile:
            self.arg_dict = pkl.load(configfile)
        self.arg_dict['action_dim'] = 4
        self.arg_dict['tau'] = 0.0005
        self.arg_dict['n'] = 5
        self.arg_dict['tname'] = 'no'
        self.DDPG = DDPGfD_priority(self.arg_dict)

    def test_select_action(self):
        rand_state = np.random.rand(self.arg_dict['state_dim'])
        rand_action = self.DDPG.select_action(rand_state)
        for i in range(self.arg_dict['action_dim']):
            self.assertGreaterEqual(rand_action[i], -1, 'random action contains component less than -1')
            self.assertLessEqual(rand_action[i], 1, 'random action contains component greater than 1')
        
        
    def test_save_and_load(self):
        self.DDPG.save('testing')
        second_ddpg = DDPGfD_priority(self.arg_dict)
        second_ddpg.load('testing')
        self.assertTrue(compare_models(self.DDPG.critic, second_ddpg.critic), 'critics different')
        self.assertTrue(compare_models(self.DDPG.actor, second_ddpg.actor), 'actors different')
        
        test_things = np.random.rand(100,8)
        for i in range(100):
            act = list(self.DDPG.select_action(test_things[i]))
            act2 = list(second_ddpg.select_action(test_things[i]))
            self.assertListEqual(act,act2)
            critic,_ = self.DDPG.grade_action(test_things[i], act)
            c2,_ = second_ddpg.grade_action(test_things[i], act2)
            self.assertEqual(critic, c2)

    def test_init(self):
        self.assertEqual(self.DDPG.use_HER, 'HER' in self.arg_dict['model'], 'Use Hindsight Experience Replay flag incorrect')
        self.assertEqual(self.DDPG.sparse_reward, 'sparse' in self.arg_dict['reward'].lower(), 'Use sparse rewards flag incorrect')
        self.assertEqual(self.DDPG.discount, self.arg_dict['discount'], 'Discount Factor incorrect')
        self.assertEqual(self.DDPG.tau, self.arg_dict['tau'], 'Tau incorrect')
        self.assertEqual(self.DDPG.n, self.arg_dict['n'], 'Number of lookahead steps incorrect')
        self.assertEqual(self.DDPG.batch_size, self.arg_dict['batch_size'], 'Batch Size incorrect')
        
    def test_update_target(self):
        # TODO: add this. not doing now because it seems like a pain and this is unlikely to be the root cause
        pass
        
if __name__ == '__main__':
    unittest.main()