from abc import ABC, abstractmethod
from copy import deepcopy

import json
import logging

from mojograsp.simcore.state import State, StateDefault
from mojograsp.simcore.action import Action, ActionDefault
from mojograsp.simcore.reward import Reward, RewardDefault
import pickle as pkl


class RecordData(ABC):
    """Record Data Abstract Base Class"""
    @abstractmethod
    def record_timestep(self):
        """
        Method is called every timestep by :func:`~mojograsp.simcore.sim_manager.SimManager` 
        to record the State, Action and Reward (if given). 
        """
        pass

    @abstractmethod
    def record_episode(self):
        """
        Method is called after every episode by :func:`~mojograsp.simcore.sim_manager.SimManager` 
        and records all of the given timesteps for an episode. 
        """
        pass

    @abstractmethod
    def save_episode(self):
        """
        Method called after record_episode(), optionally saves the previous episode output to a file.
        """
        pass

    @abstractmethod
    def save_all(self):
        """Method called after all episodes are completed, optionally condenses and saves all episodes to an output file"""
        pass


class RecordDataDefault(RecordData):
    def record_timestep(self):
        """Default Placeholder if no RecordData class is provided"""
        super().record_timestep()

    def record_episode(self):
        """Default Placeholder if no RecordData class is provided"""
        super().record_episode()

    def save_episode(self):
        """Default Placeholder if no RecordData class is provided"""
        super().save_episode()

    def save_all(self):
        """Default Placeholder if no RecordData class is provided"""
        super().save_all()


class RecordDataJSON(RecordData):
    """RecordDataJSON Class (Optional default class for recording data to json files)"""

    def __init__(self, data_path: str = None, data_prefix: str = "episode", save_all=False, save_episode=True,
                 state: State = StateDefault(), action: Action = ActionDefault(), reward: Reward = RewardDefault()):
        """
        Constructor takes in State, Action and Reward objects from the user and saves their outputs at every 
        timestep. Optionally can save each episode individually, alltogether, or not at all. State, Reward and 
        Action defaults are given if user does not define them (they will return empty dictionaries). 

        :param data_path: Desired directory path for storing resulting json files.
        :param data_prefix: Named prefix before each json file. Defaults to episode
        :param save_all: Set whether or not you would like to save all episodes into one file (Default False).
        :param save_episode: Set whether or not you would like to save each episodes into individual files (Default True).
        :param state: State object defined by user, if none given then :func:`~mojograsp.simcore.state.StateDefault` will be used. 
        :param action: Action object defined by user, if none given then :func:`~mojograsp.simcore.action.ActionDefault` will be used. 
        :param reward: Reward object defined by user, if none given then :func:`~mojograsp.simcore.reward.RewardDefault` will be used. 
        :type data_path: str
        :type data_prefix: str
        :type save_all: bool
        :type save_episode: bool
        :type state: :func:`~mojograsp.simcore.state.State` 
        :type action: :func:`~mojograsp.simcore.action.Action` 
        :type reward: :func:`~mojograsp.simcore.reward.Reward` 
        """
        if not data_path:
            logging.warn("No data path provided")
        self.data_path = data_path
        self.data_prefix = data_prefix
        self.save_all_flag = save_all
        self.save_episode_flag = save_episode
        self.state = state
        self.action = action
        self.reward = reward
        self.timestep_num = 1
        self.episode_num = 0
        self.timesteps = []
        self.episode_data = []
        self.current_episode = None
        self.episodes = {}

    def record_timestep(self):
        """
        Method called by :func:`~mojograsp.simcore.sim_manager.SimManager` every timestep. Records the user defined state, 
        action, and reward. Saves the result to a timestep dictionary which is later used to form an episode.
        """
        state_reward_dict = {}
        timestep_dict = {"number": self.timestep_num}
        if self.state:
            timestep_dict["state"] = self.state.get_state()
        if self.reward:
            timestep_dict["reward"] = self.reward.get_reward()
        if self.action:
            timestep_dict["action"] = self.action.get_action()
        self.timesteps.append(timestep_dict)
        self.timestep_num += 1

    def record_episode(self):
        """
        Method called by :func:`~mojograsp.simcore.sim_manager.SimManager` after every episode. Compiles all of the 
        timestep dictionaries into a list and adds it to the episode dicionary. 
        """
        episode = {"number": self.episode_num+1}
        episode["timestep_list"] = self.timesteps
        self.current_episode = episode

        if self.save_all_flag:
            self.episode_data.append(episode)

        self.timesteps = []
        self.timestep_num = 1
        self.episode_num += 1

    def save_episode(self):
        """
        Method called by :func:`~mojograsp.simcore.sim_manager.SimManager` after every episode. Saves the most recent
        episode dictionary to a json file. 
        """
        if self.save_episode_flag and self.data_path != None:
            file_path = self.data_path + \
                self.data_prefix + "_" + str(self.episode_num) + ".json"
            print(file_path)
            with open(file_path, 'w') as fout:
                json.dump(self.current_episode, fout, indent=4)
        self.current_episode = {}

    def save_all(self):
        """
        Method called by :func:`~mojograsp.simcore.sim_manager.SimManager` after all episodes are completed. Saves all
        episode dictionaries to a json file. 
        """

        if self.save_all_flag and self.data_path != None:
            file_path = self.data_path + \
                self.data_prefix + "_all.json"
            with open(file_path, 'w') as fout:
                self.episodes = {"episode_list": self.episode_data}
                json.dump(self.episodes, fout)

class RecordDataPKL(RecordData):
    """RecordDataJSON Class (Optional default class for recording data to json files)"""

    def __init__(self, data_path: str = None, data_prefix: str = "episode", save_all=False, save_episode=True,
                 state: State = StateDefault(), action: Action = ActionDefault(), reward: Reward = RewardDefault()):
        """
        Constructor takes in State, Action and Reward objects from the user and saves their outputs at every 
        timestep. Optionally can save each episode individually, alltogether, or not at all. State, Reward and 
        Action defaults are given if user does not define them (they will return empty dictionaries). 

        :param data_path: Desired directory path for storing resulting json files.
        :param data_prefix: Named prefix before each json file. Defaults to episode
        :param save_all: Set whether or not you would like to save all episodes into one file (Default False).
        :param save_episode: Set whether or not you would like to save each episodes into individual files (Default True).
        :param state: State object defined by user, if none given then :func:`~mojograsp.simcore.state.StateDefault` will be used. 
        :param action: Action object defined by user, if none given then :func:`~mojograsp.simcore.action.ActionDefault` will be used. 
        :param reward: Reward object defined by user, if none given then :func:`~mojograsp.simcore.reward.RewardDefault` will be used. 
        :type data_path: str
        :type data_prefix: str
        :type save_all: bool
        :type save_episode: bool
        :type state: :func:`~mojograsp.simcore.state.State` 
        :type action: :func:`~mojograsp.simcore.action.Action` 
        :type reward: :func:`~mojograsp.simcore.reward.Reward` 
        """
        if not data_path:
            logging.warn("No data path provided")
        self.data_path = data_path
        self.data_prefix = data_prefix
        self.save_all_flag = save_all
        self.save_episode_flag = save_episode
        self.state = state
        self.action = action
        self.reward = reward
        self.timestep_num = 1
        self.episode_num = 0
        self.timesteps = []
        self.episode_data = []
        self.current_episode = None
        self.episodes = {}
        self.eval_num = 0

    def record_timestep(self):
        """
        Method called by :func:`~mojograsp.simcore.sim_manager.SimManager` every timestep. Records the user defined state, 
        action, and reward. Saves the result to a timestep dictionary which is later used to form an episode.
        """
        state_reward_dict = {}
        timestep_dict = {"number": self.timestep_num}
        if self.state:
            timestep_dict["state"] = self.state.get_state()
        if self.reward:
            timestep_dict["reward"] = self.reward.get_reward()
        if self.action:
            timestep_dict["action"] = self.action.get_action()
        self.timesteps.append(timestep_dict)
        self.timestep_num += 1

    def record_episode(self,episode_type='train'):
        """
        Method called by :func:`~mojograsp.simcore.sim_manager.SimManager` after every episode. Compiles all of the 
        timestep dictionaries into a list and adds it to the episode dicionary. 
        """
        episode = {"number": self.episode_num+1, "hand name":self.state.get_hand_name()}
        episode["timestep_list"] = self.timesteps
        
        self.current_episode = episode

        if self.save_all_flag:
            self.episode_data.append(episode)

        self.timesteps = []
        self.timestep_num = 1
        if episode_type == 'train':
            self.episode_num += 1
            self.eval_num = self.episode_num

    def save_episode(self,episode_type='train', filename=None, use_reward_name=False):
        """
        Method called by :func:`~mojograsp.simcore.sim_manager.SimManager` after every episode. Saves the most recent
        episode dictionary to a pkl file. 
        """
        if self.save_episode_flag and self.data_path != None:
            if episode_type == 'test':
                if filename is not None:
                    file_path = self.data_path + "Test/"+ filename + '_' + str(self.eval_num) + '.pkl'
                elif use_reward_name:
                    name = self.state.get_name()
                    file_path = self.data_path + "Test/"+ name + '_' + str(self.eval_num) + '.pkl'
                else:
                    file_path = self.data_path + \
                        "Test/Evaluation_episode_" + str(self.eval_num) + ".pkl" 
                    
                self.eval_num +=1
                # print('save episode evaluated', self.eval_num)
            elif episode_type == 'train':
                if filename is None:
                    file_path = self.data_path + 'Train/' + \
                        self.data_prefix + "_" + str(self.episode_num) + ".pkl"
                else:
                    file_path = self.data_path + 'Train/' +  filename
            elif episode_type == 'eval':
                if filename is None:
                    file_path = self.data_path + 'Eval/' + \
                        self.data_prefix + "_" + str(self.eval_num) + ".pkl"
                else:
                    file_path = self.data_path + 'Eval/' +  filename
                self.eval_num +=1
                # print('save episode evaluated', self.eval_num)

            print(file_path, len(self.current_episode['timestep_list']))
            
            with open(file_path, 'wb') as fout:
                pkl.dump(self.current_episode, fout)
        self.current_episode = {}

    def save_all(self):
        """
        Method called by :func:`~mojograsp.simcore.sim_manager.SimManager` after all episodes are completed. Saves all
        episode dictionaries to a pkl file. 
        """
        print('calling save all')
        if self.save_all_flag and self.data_path != None:
            file_path = self.data_path + \
                self.data_prefix + "_all.pkl"
            with open(file_path, 'wb') as fout:
                self.episodes = {"episode_list": self.episode_data}
                pkl.dump(self.episodes, fout)
        

    def clear(self):
        self.current_episode = {}
        self.timesteps = []
        self.timestep_num = 1

class RecordDataRLPKL(RecordDataPKL):

    def __init__(self, data_path: str = None, data_prefix: str = "episode", save_all=False, save_episode=True,
                 state: State = StateDefault(), action: Action = ActionDefault(), reward: Reward = RewardDefault(), controller = None):
        RecordDataPKL.__init__(self, data_path, data_prefix, save_all, save_episode, state, action, reward)
        self.controller = controller
        
        
    def record_timestep(self):
        """
        Method called by :func:`~mojograsp.simcore.sim_manager.SimManager` every timestep. Records the user defined state, 
        action, and reward. Saves the result to a timestep dictionary which is later used to form an episode.
        """
        state_reward_dict = {}
        timestep_dict = {"number": self.timestep_num}
        if self.state:
            temp = self.state.get_state()
            exclude_keys = ['previous_state', 'image']
            timestep_dict["state"] = {k: temp[k] for k in set(list(temp.keys())) - set(exclude_keys)}
        if self.reward:
            timestep_dict["reward"] = self.reward.get_reward()
        if self.action:
            timestep_dict["action"] = self.action.get_action()
        if self.controller:
            timestep_dict["control"] = self.controller.get_network_outputs(self.state.get_state())
        self.timesteps.append(timestep_dict)
        self.timestep_num += 1
        
