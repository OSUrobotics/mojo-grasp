from collections import namedtuple, deque
from random import shuffle, randint
from copy import deepcopy
from dataclasses import dataclass
import csv
import numpy as np


@dataclass
class Timestep:
    episode: int
    phase: str
    wall_time: float
    sim_time: float
    timestep: int
    current_state: list
    action: list
    reward: float
    next_state: list
    # done: bool

    def __iter__(self):
        '''Allows us to iterate through the timestep dataclass as a list of strings'''
        timestep_list = [str(self.episode), str(self.phase), str(
            self.wall_time), str(self.sim_time), str(self.timestep)]
        timestep_list.extend([str(x) for x in self.current_state])
        timestep_list.extend([str(x) for x in self.action])
        timestep_list.append(self.reward)
        if self.next_state:
            timestep_list.extend([str(x) for x in self.next_state])
        else:
            timestep_list.append("")
        # timestep_list.append(self.done)
        return iter(timestep_list)


class ReplayBuffer:
    def __init__(self, episodes_file=None, buffer_size=10000):
        '''Initializes episode file being used and the replay buffer structure'''
        self.episodes_file = episodes_file
        self.buffer_size = buffer_size

        # deque data structure deletes oldest entry in array once buffer_size is exceeded
        self.current_buffer = deque(maxlen=buffer_size)

        # creates our replay buffer from episodes_file
        if self.episodes_file:
            self.create_replay_buffer()
        self.n_step_indices = []
        self.n = 5
        for num in range(self.buffer_size - self.n):
            self.n_step_indices.append(np.arange(num, num + self.n))

    def create_replay_buffer(self):
        '''Creates replay buffer from the given episodes_file in the __init__. Reads csv file
           row by row and creates timestep object for each before storing them into the current_buffer.'''
        if self.episodes_file:
            current_episode = 0
            num_action_vars = 0
            num_state_vars = 0
            # open csv file with timesteps
            with open(self.episodes_file) as csv_file:
                reader = csv.reader(csv_file)
                for row in reader:
                    # check if there is a header and we are entering a new episode
                    if ":" in row[0]:
                        current_episode = int(row[0].split(":")[1])
                        num_state_vars = int(row[5].split(":")[1])
                        num_action_vars = int(row[6].split(":")[1])
                    # otherwise we construct our Timestep object and add it to the replay buffer
                    else:
                        # get state, action and reward values from row and cast to floats
                        current_state = list(
                            map(float, row[4:num_state_vars+4]))
                        action = list(
                            map(float, row[num_state_vars+4:num_action_vars+num_state_vars+4]))
                        # print("State:", current_state)
                        # print("Ac", action)
                        # current_state = list(map(float,row[4:num_state_vars]))
                        # action = list(map(float,row[num_state_vars:num_action_vars+num_state_vars]))
                        reward = None
                        if row[-1] != "":
                            try:
                                reward = float(row[-1])
                            except ValueError:
                                reward = float(
                                    row[-1].strip('][').split(',')[0])
                        # fill in previous timesteps next state based on the new timestep current state
                        if len(self.current_buffer) >= 1:
                            self.current_buffer[-1].next_state = current_state
                        # create Timestep object and add to buffer
                        new_timestep = Timestep(current_episode, row[0], float(row[1]), float(row[2]), int(row[3]),
                                                current_state, action, reward, None)
                        self.current_buffer.append(new_timestep)

    def add_episode(self, new_episode, ep_num):
        '''Adds all of the timesteps from a given episode into the replay buffer.
           @param new_episode - Episode object containing all the timesteps in an episode'''
        for i in new_episode.data.values():
            self.add_timestep(i, ep_num)

    def add_episode_filter_phases(self, new_episode, phases=[]):
        '''Adds in all timesteps from certain phases into the replay buffer.
           @param new_episode - Episode object containing all the timesteps in an episode
           @param phases - list of phase names whose correspoding timesteps you wish to add to the replay buffer'''
        for i in new_episode.data.values():
            timestep_data = i.get_full_timestep()
            if timestep_data['phase'] in phases:
                timestep = Timestep(new_episode.episode_number, timestep_data['phase'], timestep_data["wall_time"],
                                    timestep_data["sim_time"], timestep_data["timestep"],
                                    timestep_data["state"], timestep_data["action"], timestep_data["reward"], None)
                if len(self.current_buffer) >= 1:
                    self.current_buffer[-1].next_state = timestep_data["state"]
                self.current_buffer.append(timestep)

    def add_timestep(self, new_timestep, episode_number):
        '''Adds a single timestep into the replay buffer.
           @param new_timestep - Timestep object
           @param episode_number - corresponding episode number for the timestep TODO: shouldnt be done here, should be passed into timestep'''
        timestep_data = new_timestep.get_full_timestep()
        timestep = Timestep(episode_number, timestep_data['phase'], timestep_data["wall_time"], timestep_data["sim_time"], timestep_data["timestep"],
                            timestep_data["state"], timestep_data["action"], timestep_data["reward"], None)
        if len(self.current_buffer) > 1:
            self.current_buffer[-1].next_state = timestep_data["state"]
        self.current_buffer.append(timestep)

    def remove_episode(self, episode_number):
        '''Removes all timesteps within the corresponding episode number from the replay buffer
           @param episode_number - episode number you wish to remove'''
        for i in list(self.current_buffer):
            if i.episode == episode_number:
                self.current_buffer.remove(i)

    def remove_timestep(self, episode_number, timestep_number):
        '''Removes a timestep based on the timestep number and episode number.
           @param episode_number - episode the timestep is in
           @param timestep_number - timestep number you wish to remove'''
        for i in list(self.current_buffer):
            if i.episode == episode_number and i.timestep == timestep_number:
                self.current_buffer.remove(i)

    def get_episode(self, episode_number):
        '''Returns list of timesteps from a given episode number.
           @param episode_number - episode number you wish to get timesteps of
           @return - list of timestep data class objects'''
        timestep_list = []
        for i in list(self.current_buffer):
            if i.episode == episode_number:
                timestep_list.append(i)
        return timestep_list

    def get_episode_batch(self, num_episodes, n=None, start_timestep=None, end_timestep=None):
        """
        Returns samples of entire episodes, in corrct timestep sequence
        :param num_episodes: Number of episodes to sample
        :param n: n-step (To calculate ceiling of timestep to sample per episode)
        :param start_timestep: Timestep to start sampling from, per episode
        :param end_timestep: Timestep to end sampling from, per episode
        :return: list of timestep dataclass objects
        """
        return [full_episode for full_episode in self.current_buffer if full_episode.episode == 1]
        pass

    def get_timestep(self, episode_number, timestep_number):
        '''Returns a timestep from a given episode number.
           @param episode_number - episode number you want to get timestep from 
           @param timestep_number - timestep number from episode
           @return - Timestep data class object'''
        for i in list(self.current_buffer):
            if i.episode == episode_number and i.timestep == timestep_number:
                return i

    def get_random_timestep_sample(self, num_timesteps):
        '''Returns a list of timesteps from random episodes and timestep numbers
           @param num_timesteps - number of timesteps to return
           @return - List of timestep data class object'''
        timestep_list = []
        if len(self.current_buffer) >= num_timesteps:
            for i in range(num_timesteps):
                timestep_list.append(
                    self.current_buffer[randint(0, len(self.current_buffer)-1)])
        return timestep_list

    def get_last_timestep_from_episode(self, episode_num):
        for i in list(self.current_buffer):
            if i.episode == episode_num:
                last_timestep = i.timestep
            if i.episode > episode_num:
                break
        return last_timestep

    def get_between_timestep_random_sample(self, num_timesteps, start_timestep, end_timestep=None):
        timestep_list = []
        if len(self.current_buffer) >= num_timesteps:
            for i in range(num_timesteps):
                done = False
                while not done:
                    random_timestep_index = randint(
                        0, len(self.current_buffer) - 1)
                    last_step = self.get_last_timestep_from_episode(
                        self.current_buffer[random_timestep_index].episode)
                    if 'move' in self.current_buffer[random_timestep_index].phase and self.current_buffer[random_timestep_index].timestep is not last_step:
                        done = True
                timestep_list.append(
                    self.current_buffer[random_timestep_index])
        return timestep_list

    def get_random_episode_sample(self, ceil, num_ep=1):
        """
        Samples entire episodes (in order of their timesteps) until they reach ceiling timestep value of that episode
        :param ceil: Number of timesteps to subtract from final timestep (n-step)
        :param num_ep: Number of episodes to sample
        :return: episodes_list: A list of all timesteps in order of episodes sampled
        """
        all_eps = []
        if len(self.current_buffer) >= num_ep:
            # get a list of random episode numbers to sample
            episode_indices = []
            while len(episode_indices) < num_ep:
                # print("Last episode in buffer:", self.current_buffer[-1].episode)
                rand_num = randint(
                    self.current_buffer[0].episode, self.current_buffer[-1].episode)
                if rand_num in episode_indices:
                    continue
                found_ts = False
                i = 0
                try:
                    while not found_ts:
                        ts = self.current_buffer[i]
                        if ts.episode == rand_num and ts.timestep == 43:
                            found_ts = True
                        i += 1
                    if not found_ts:
                        continue
                except IndexError:
                    continue
                episode_indices.append(rand_num)

            for index in episode_indices:
                buffer_episode = [
                    ts for ts in self.current_buffer if index == ts.episode]
                # Find all the timesteps of all the episodes, in order, append to list
                curr_ep = buffer_episode[0].episode
                episodes_list = []
                # Iterate through buffer
                for i in list(buffer_episode):
                    # append timestep to list if current episode matches an episode index from randomly generated list and fullfills other criteria
                    if 'move' in i.phase:
                        last_step = self.get_last_timestep_from_episode(
                            curr_ep) - ceil
                        if i.timestep <= last_step:
                            episodes_list.append(i)
                all_eps.append(episodes_list)

        # if num_ep == 1:
        #     return episodes_list
        return all_eps

    def sample_batch_nstep(self, batch_size):
        """

        :param batch_size:
        :param n_step:
        :return:
        """
        all_eps = []
        if len(self.current_buffer) >= batch_size:
            # print(len(self.current_buffer), batch_size)
            # get a list of random episode numbers to sample
            episode_indices = []
            state_arr, action_arr, next_state_arr, reward_arr = [], [], [], []
            select_from_episodes = np.asarray(
                range(self.current_buffer[0].episode, self.current_buffer[-1].episode))
            while len(episode_indices) < batch_size:
                ep_state, ep_action, ep_next_state, ep_reward = [], [], [], []
                rand_num = np.random.choice(
                    select_from_episodes, replace=False)
                for all_ts in self.current_buffer:
                    if rand_num == all_ts.episode and 'move' in all_ts.phase:
                        ep_state.append(np.asarray(all_ts.current_state))
                        ep_action.append(np.asarray(all_ts.action))
                        ep_next_state.append(np.asarray(all_ts.next_state))
                        ep_reward.append(np.asarray(all_ts.reward))
                    if all_ts.episode > rand_num:
                        break

                if not len(ep_state):
                    continue
                temp_state, temp_action, temp_next_state, temp_reward = np.asarray(
                    ep_state), np.asarray(ep_action), np.asarray(ep_next_state), np.asarray(ep_reward)

                traj_arr = self.n_step_indices[:(len(temp_reward) - self.n)]
                for row in traj_arr:
                    state_arr.append(temp_state[row])
                    action_arr.append(temp_action[row])
                    next_state_arr.append(temp_next_state[row])
                    reward_arr.append(temp_reward[row])
                episode_indices.append(rand_num)

        return (torch.FloatTensor(state_arr).to(device), torch.FloatTensor(action_arr).to(device),
                torch.FloatTensor(next_state_arr).to(device), torch.FloatTensor(reward_arr).to(device))

    def get_recent_timestep_sample(self, num_timesteps):
        '''Returns a list of n timesteps from most recently added timesteps
           @param num_timesteps - number of timesteps to return
           @return - List of timestep data class object'''
        timestep_list = []
        if len(self.current_buffer) >= num_timesteps:
            for i in range(num_timesteps):
                timestep_list.append(self.current_buffer[-(i+1)])
        return timestep_list

    def get_oldest_timestep_sample(self, num_timesteps):
        '''Returns a list of n timesteps from most oldest added timesteps
           @param num_timesteps - number of timesteps to return
           @return - List of timestep data class object'''
        timestep_list = []
        if len(self.current_buffer) >= num_timesteps:
            for i in range(num_timesteps):
                timestep_list.append(self.current_buffer[i])
        return timestep_list

    def save_replay_buffer(self, file_name):
        '''Saves contents of the replay buffer to a csv file 
           @param file_name - Name of replay buffer file
           @return - List of timestep data class object'''
        current_episode = -1
        num_action_vars = 0
        num_state_vars = 0
        num_state_vars_next = 0
        with open(file_name, 'w', newline='') as csv_file:
            replay_write = csv.writer(
                csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for i in self.current_buffer:
                if i.episode != current_episode:
                    current_episode = i.episode
                    num_action_vars = len(i.action)
                    num_state_vars = len(i.current_state)
                    if i.next_state:
                        num_state_vars_next = len(i.next_state)
                    else:
                        num_state_vars_next = 0
                    header_text = ['Episode:'+str(current_episode), 'Phase', 'WallTime', 'SimTime', 'TimeStep',
                                   'State:' +
                                   str(num_state_vars), 'Action:' +
                                   str(num_action_vars),
                                   'Reward:1', 'NextState:'+str(num_state_vars_next)]
                    replay_write.writerow(header_text)
                    replay_write.writerow(i)
                else:
                    replay_write.writerow(i)
