from mojograsp.simcore.record_data import RecordDataRLPKL
from mojograsp.simcore.state import State, StateDefault
from mojograsp.simcore.action import Action, ActionDefault
from mojograsp.simcore.reward import Reward, RewardDefault
import pickle as pkl

class MultiprocessRecordData(RecordDataRLPKL):
    def __init__(self, Record_id, data_path: str = None, data_prefix: str = "episode", save_all=False, save_episode=True,
                 state: State = StateDefault(), action: Action = ActionDefault(), reward: Reward = RewardDefault(), controller = None):
        super().__init__(data_path=data_path, data_prefix=data_prefix, save_all=save_all, save_episode=save_episode,state=state,action=action, reward=reward, controller=controller)
        self.record_id = Record_id

    def save_episode(self,evaluated='train', filename=None, use_reward_name=False):
        """
        Method called by :func:`~mojograsp.simcore.sim_manager.SimManager` after every episode. Saves the most recent
        episode dictionary to a pkl file. 
        """
        if self.save_episode_flag and self.data_path != None:
            if evaluated == 'test':
                if filename is None:
                    print(self.data_path)
                    print(self.record_id)
                    file_path = self.data_path + "Test/"+ self.record_id + str(self.eval_num) + '.pkl'
                elif use_reward_name:
                    name = self.state.get_name()
                    file_path = self.data_path + "Test/"+ self.record_id + name + str(self.eval_num) + '.pkl'
                else:
                    file_path = self.data_path + \
                        "Test/" + self.record_id + "Evaluation_episode_" + str(self.eval_num) + ".pkl" 
                    
                self.eval_num +=1
                print('save episode evaluated', self.eval_num)
            else:
                if filename is None:
                    file_path = self.data_path + 'Train/' + \
                        self.record_id + self.data_prefix + "_" + str(self.episode_num) + ".pkl"
                else:
                    file_path = self.data_path + 'Train/' + self.record_id + filename
            print(file_path)
            
            with open(file_path, 'wb') as fout:
                pkl.dump(self.current_episode, fout)
        self.current_episode = {}