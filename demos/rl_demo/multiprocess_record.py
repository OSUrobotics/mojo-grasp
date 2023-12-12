from mojograsp.simcore.record_data import RecordDataRLPKL
from mojograsp.simcore.state import State, StateDefault
from mojograsp.simcore.action import Action, ActionDefault
from mojograsp.simcore.reward import Reward, RewardDefault
import pickle as pkl

class MultiprocessRecordData(RecordDataRLPKL):
    def __init__(self, Record_id: list, data_path: str = None, data_prefix: str = "episode", save_all=False, save_episode=True,
                 state: State = StateDefault(), action: Action = ActionDefault(), reward: Reward = RewardDefault(), controller = None):
        super().__init__(data_path=data_path, data_prefix=data_prefix, save_all=save_all, save_episode=save_episode,state=state,action=action, reward=reward, controller=controller)
        self.num_threads = float(Record_id[1])
        self.my_thread = float(Record_id[0])


    def save_episode(self,evaluated='train', filename=None, use_reward_name=False):
        """
        Method called by :func:`~mojograsp.simcore.sim_manager.SimManager` after every episode. Saves the most recent
        episode dictionary to a pkl file. 
        """
        print(self.eval_num,self.num_threads, self.my_thread)
        print(int(self.eval_num*self.num_threads+self.my_thread))
        if self.save_episode_flag and self.data_path != None:
            if evaluated == 'test':
                if filename is None:
                    
                    file_path = self.data_path + "Test/Episode_"+ str(int(self.eval_num*self.num_threads+self.my_thread)) + '.pkl'
                elif use_reward_name:
                    name = self.state.get_name()
                    file_path = self.data_path + "Test/"+ name + str(int(self.eval_num*self.num_threads+self.my_thread)) + '.pkl'
                else:
                    file_path = self.data_path + \
                        "Test/Episode_" + filename + ".pkl"
                    
                self.eval_num +=1
                print('save episode evaluated', self.eval_num*self.num_threads+self.my_thread)
            else:
                if filename is None:
                    file_path = self.data_path + 'Train/' + \
                        self.data_prefix + "_" + str(int(self.eval_num*self.num_threads+self.my_thread)) + ".pkl"
                else:
                    file_path = self.data_path + 'Train/' + filename + ".pkl"
            print(file_path)
            
            with open(file_path, 'wb') as fout:
                pkl.dump(self.current_episode, fout)
        self.current_episode = {}