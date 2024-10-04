from mojograsp.simcore.record_data import RecordDataRLPKL
from mojograsp.simcore.state import State, StateDefault
from mojograsp.simcore.action import Action, ActionDefault
from mojograsp.simcore.reward import Reward, RewardDefault
import pickle as pkl
def getitem_for(d, key):
    for level in key:
        d = d[level]
    return d


class MultiprocessRecordData(RecordDataRLPKL):
    reduced_keys={'Start Pos':(0,'state','obj_2','pose',0),'End Pos':(0,'state','obj_2','pose',0),
                      'Goal Position': (0, 'state','goal_pose','goal_position'), 'Start Distance': (0,'reward','distance_to_goal'),
                      'End Distance': (-1, 'reward','distance_to_goal'), 'Max Distance': ('max','reward','distance_to_goal'),
                      'End Orientation': (-1,'reward','object_orientation',2), 'Goal Orientation': (0, 'state','goal_pose','goal_orientation'),
                      'Slide Sum':('sum','reward','distance_to_goal'), 'Rotate Sum':('abssum',('reward','object_orientation',2),('state','goal_pose','goal_orientation')),
                      'Finger Sum':('maxsum',('reward','f1_dist'),('reward','f2_dist'))}

    def __init__(self, Record_id: list, data_path: str = None, data_prefix: str = "episode", save_all=False, save_episode=True,
                 state: State = StateDefault(), action: Action = ActionDefault(), reward: Reward = RewardDefault(), controller = None):
        super().__init__(data_path=data_path, data_prefix=data_prefix, save_all=save_all, save_episode=save_episode,state=state,action=action, reward=reward, controller=controller)
        self.num_threads = float(Record_id[1])
        self.my_thread = float(Record_id[0])

    def record_episode(self, episode_type='train', frictionList = None, contactList = None):
        super().record_episode(episode_type)
        self.current_episode["frictionList"] = frictionList
        self.current_episode["contactList"] = contactList
        # print("Friction list in record is :", frictionList)


    def save_episode(self,evaluated='train', filename=False, hand_type=None):
        """
        Method called by :func:`~mojograsp.simcore.sim_manager.SimManager` after every episode. Saves the most recent
        episode dictionary to a pkl file. 
        """
        # print(self.eval_num,self.num_threads, self.my_thread)
        # print(int(self.eval_num*self.num_threads+self.my_thread))
        if self.save_episode_flag and self.data_path != None:
            if evaluated == 'test':
                if hand_type is None:
                    file_path = self.data_path + "Test/Episode_" + str(int(self.eval_num*self.num_threads+self.my_thread)) + '.pkl'
                elif hand_type=='A':
                    file_path = self.data_path + "Eval_A/Episode_" + str(int(self.eval_num*self.num_threads+self.my_thread)) + '.pkl'
                elif hand_type=='B':
                    file_path = self.data_path + "Eval_B/Episode_" + str(int(self.eval_num*self.num_threads+self.my_thread)) + '.pkl'
                else:
                    file_path = self.data_path + "Eval_" + hand_type+"/Episode_" + str(int(self.eval_num*self.num_threads+self.my_thread)) + '.pkl'
                self.eval_num +=1
                print('save episode evaluated', self.eval_num*self.num_threads+self.my_thread, hand_type)
            elif evaluated == 'asterisk':
                # print('hu')
                if hand_type is None:
                    file_path = self.data_path + "Test/Episode_" + str(int(self.eval_num*self.num_threads+self.my_thread)) + '.pkl'
                elif hand_type=='A':
                    file_path = self.data_path + "Ast_A/Episode_" + str(int(self.eval_num*self.num_threads+self.my_thread)) + '.pkl'
                elif hand_type=='B':
                    file_path = self.data_path + "Ast_B/Episode_" + str(int(self.eval_num*self.num_threads+self.my_thread)) + '.pkl'
                elif hand_type=='A_A':
                    file_path = self.data_path + "Full_Ast_A/Episode_" + str(int(self.eval_num*self.num_threads+self.my_thread)) + '.pkl'
                elif hand_type=='B_B':
                    file_path = self.data_path + "Full_Ast_B/Episode_" + str(int(self.eval_num*self.num_threads+self.my_thread)) + '.pkl'
                self.eval_num +=1
            else:
                if hand_type is None:
                    file_path = self.data_path + 'Train/' + \
                        self.data_prefix + "_" + str(int(self.eval_num*self.num_threads+self.my_thread)) + ".pkl"
                else:
                    file_path = self.data_path + 'Train/' + filename + ".pkl"
            
            with open(file_path, 'wb') as fout:
                pkl.dump(self.current_episode, fout)
        self.current_episode = {}

    def record_test_round(self):
        '''
        records a reduced test data. assumes you aren't saving training data'''
        print('TESTING RECORD TEST ROUND')
        save_dict = {"number":self.episode_num+1}
        for key,sequence in MultiprocessRecordData.reduced_keys.items():
            if type(sequence[0]) is int:
                save_dict[key] = getitem_for(self.timesteps[sequence[0]],sequence[1:])
            elif sequence[0] == 'max':
                save_dict[key] = max([getitem_for(i,sequence[1:]) for i in self.timesteps])
            elif sequence[0] == 'sum':
                save_dict[key] = sum([getitem_for(i,sequence[1:]) for i in self.timesteps])
                print('sum thing', key)
            elif sequence[0] == 'abssum':
                save_dict[key] = sum([abs(getitem_for(i,sequence[1]) - getitem_for(i,sequence[2]) )for i in self.timesteps])
                print('abs sum thing', key)
            elif sequence[0] == 'maxsum':
                save_dict[key] = sum([max(getitem_for(i,sequence[1]),getitem_for(i,sequence[2])) for i in self.timesteps])
                print('max sum thing', key)
            else:
                raise NotImplemented('unknown keyworkd')
        self.current_episode = save_dict
        self.timesteps = []
        self.timestep_num = 1
        self.episode_num += 1
