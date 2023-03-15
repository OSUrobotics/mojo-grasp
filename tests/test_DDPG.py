# project/test.py

import unittest
from mojograsp.simcore.DDPGfD import DDPGfD_priority, simple_normalize
import pickle as pkl
import numpy as np
import torch
import json

def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            return False
    return True
        

class FakeReplay():
    def __init__(self,data):
        self.data = data
        self.len = 10000000000
    
    def __len__(self):
        return self.len

    def sample_rollout(self, BATCH_SIZE, ROLLOUT_SIZE):
        tuples = []
        weights = []
        indexes = []
        for i,timestep in enumerate(self.data):
            temp = (timestep['state'],timestep['action'],timestep['reward'],self.data[i+1]['state'],0,0,i)
            tuples.append([temp]*ROLLOUT_SIZE)
            indexes.append([i] * ROLLOUT_SIZE)
            weights.append([1] * ROLLOUT_SIZE)
            if i+1 >= BATCH_SIZE:
                return tuples, weights, indexes

class TestDDPG(unittest.TestCase):

    def setUp(self):
        with open('./test_configs/experiment_config.json','r') as configfile:
            self.arg_dict = json.load(configfile)
        self.arg_dict['action_dim'] = 4
        self.arg_dict['tau'] = 0.0005
        self.arg_dict['n'] = 5
        self.arg_dict['tname'] = 'no'
        self.DDPG = DDPGfD_priority(self.arg_dict)
        with open('./test_configs/test_episode.pkl','rb') as datafile:
            temp = pkl.load(datafile)
            self.episode_data = temp['timestep_list']

    def test_select_action(self):
        rand_state = self.episode_data[0]['state']
        rand_action = self.DDPG.select_action(rand_state)
        for i in range(self.arg_dict['action_dim']):
            self.assertGreaterEqual(rand_action[i], -1, 'random action contains component less than -1')
            self.assertLessEqual(rand_action[i], 1, 'random action contains component greater than 1')
        self.assertEqual(type(rand_action),np.ndarray,'selected action not a numpy array')        
        
    def test_save_and_load(self):
        self.DDPG.save('testing')
        second_ddpg = DDPGfD_priority(self.arg_dict)
        second_ddpg.load('testing')
        self.assertTrue(compare_models(self.DDPG.critic, second_ddpg.critic), 'critics different')
        self.assertTrue(compare_models(self.DDPG.actor, second_ddpg.actor), 'actors different')
        
        for i in range(100):
            act = list(self.DDPG.select_action(self.episode_data[i]['state']))
            act2 = list(second_ddpg.select_action(self.episode_data[i]['state']))
            self.assertListEqual(act,act2)
            critic,_ = self.DDPG.grade_action(self.episode_data[i]['state'], act)
            c2,_ = second_ddpg.grade_action(self.episode_data[i]['state'], act2)
            self.assertEqual(critic, c2)

    def test_init(self):
        self.assertEqual(self.DDPG.USE_HER, 'HER' in self.arg_dict['model'], 'Use Hindsight Experience Replay flag incorrect')
        self.assertEqual(self.DDPG.SAMPLING_STRATEGY, self.arg_dict['sampling'], 'Use sparse rewards flag incorrect')
        self.assertEqual(self.DDPG.DISCOUNT, self.arg_dict['discount'], 'Discount Factor incorrect')
        self.assertEqual(self.DDPG.TAU, self.arg_dict['tau'], 'Tau incorrect')
        self.assertEqual(self.DDPG.ROLLOUT_SIZE, self.arg_dict['rollout_size'], 'Number of lookahead steps incorrect')
        self.assertEqual(self.DDPG.BATCH_SIZE, self.arg_dict['batch_size'], 'Batch Size incorrect')
        
    def test_update_target(self):
        # TODO: add this. not doing now because it seems like a pain and this is unlikely to be the root cause
        pass
    
    def test_build_reward(self):
        reward_versions = ['Sparse','Distance','Distance + Finger']
        original_version = self.DDPG.REWARD_TYPE
        for version in reward_versions:
            self.DDPG.REWARD_TYPE = version
            for timestep in self.episode_data:
                calculated_reward = self.DDPG.build_reward(timestep['reward'])
                if version == 'Sparse':
                    desired_reward = timestep['reward']['distance_to_goal'] < 0.002
                if version == 'Distance':
                    desired_reward = max(-timestep['reward']['distance_to_goal'],-1)
                if version == 'Distance + Finger':
                    desired_reward = max(-timestep['reward']['distance_to_goal']- max(timestep['reward']['f1_dist'], timestep['reward']['f2_dist'])/5,-1)
                self.assertEqual(desired_reward, calculated_reward, 'reward built incorrectly')
        self.DDPG.REWARD_TYPE = original_version
        
    def test_build_state(self):
        # only tests state most commonly used (object pose, finger pose, goal pose)
        old_state_list = self.DDPG.state_list
        new_state_list = ['op','ftp','gp']
        self.DDPG.state_list = new_state_list
        for timestep in self.episode_data:
            state_vector = []
            state_vector.extend(timestep['state']['obj_2']['pose'][0][0:2])
            state_vector.extend(timestep['state']['f1_pos'][0:2])
            state_vector.extend(timestep['state']['f2_pos'][0:2])
            state_vector.extend(timestep['state']['goal_pose']['goal_pose'])
            ddpg_state = self.DDPG.build_state(timestep['state'])
            self.assertListEqual(ddpg_state, state_vector, 'state build incorrect')
        self.DDPG.state_list = old_state_list
        
    def test_collect_batch(self):
        # Not sure how to test that all these things are actually what we say they are. there are too many of the damn things
        replay_buffer = FakeReplay(self.episode_data)
        state, action, next_state, reward, rollout_reward, rollout_discount, last_state, trimmed_weight, trimmed_idxs, expert_status = self.DDPG.collect_batch(replay_buffer)
    
    def test_train(self):
        # this one is also weird for testing
        pass
    
    def test_normalize(self):
        example_maxes = np.array([1,5,0.2,0.6,44,-3])
        example_mins = np.array([0,-5,0.1,-0.4,-2,-5])
        test_in = np.array([0.5,-2,0.19,0.1,0,-2])
        solution = np.array([0, -0.4, 0.8, 0.0, -0.9130434782608696, 2])
        test_answer = simple_normalize(test_in, example_mins, example_maxes)
        for i in range(6):
            self.assertAlmostEqual(solution[i], test_answer[i], places=3)
        
    def test_grade_action(self):
        state_in = self.episode_data[0]['state']
        action_out = self.DDPG.select_action(state_in)
        grade, action_gradient = self.DDPG.grade_action(state_in, action_out)
        self.assertEqual(type(grade),np.ndarray,'grade is not a numpy array')
        self.assertEqual(type(action_gradient),np.ndarray,'action gradient is not a numpy array')
        
if __name__ == '__main__':
    unittest.main()