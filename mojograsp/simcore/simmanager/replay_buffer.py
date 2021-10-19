from collections import namedtuple, deque
from random import shuffle, randint
from copy import deepcopy
from dataclasses import dataclass
from mojograsp.simcore.simmanager.record_timestep import RecordTimestep
from mojograsp.simcore.simmanager.record_episode import RecordEpisode
import csv 

@dataclass
class Timestep:
    episode: int
    wall_time: float
    sim_time: float
    timestep: int
    current_state: list
    action: list
    reward: float
    next_state: list

class ReplayBuffer():

    def __init__(self, episodes_file=None, buffer_size=1000):
        self.episodes_file = episodes_file
        self.buffer_size = buffer_size

        #deque data structure deletes oldest entry in array once buffer_size is exceeded
        self.current_buffer = deque(maxlen=buffer_size)
        self.shuffled_buffer = deque(maxlen=buffer_size)


        #creates our replay buffer from episodes_file
        self.create_replay_buffer()
         
    def create_replay_buffer(self):
        '''Creates replay buffer from the given episodes_file in the __init__. Reads csv file
           row by row and creates timestep object for each before storing them into the current_buffer.'''
        current_episode = 0
        num_action_vars = 0
        num_state_vars = 0
        #open csv file with timesteps
        with open(self.episodes_file) as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                #check if there is a header and we are entering a new episode
                if ":" in row[0]:
                    current_episode = int(row[0].split(":")[1])
                    num_state_vars = int(row[4].split(":")[1])
                    num_action_vars = int(row[5].split(":")[1])
                #otherwise we construct our Timestep object and add it to the replay buffer
                else:
                    #get state, action and reward values from row and cast to floats
                    current_state = list(map(float,row[3:num_state_vars+3]))
                    action = list(map(float,row[num_state_vars+3:num_action_vars+num_state_vars+3]))
                    reward = None
                    if row[-1] != "":
                        reward = float(row[-1])
                    #fill in previous timesteps next state based on the new timestep current state
                    if len(self.current_buffer) > 1:
                        self.current_buffer[-1].next_state = current_state
                    #create Timestep object and add to buffer
                    new_timestep = Timestep(current_episode,float(row[0]),float(row[1]),int(row[2]),
                                   current_state, action,reward,None)
                    self.current_buffer.append(new_timestep)


    def add_episode(self, new_episode):
        for i in new_episode.data.values():
            self.add_timestep(i, new_episode.episode_number)

    def add_timestep(self, new_timestep, episode_number):
        timestep_data = new_timestep.get_full_timestep()
        timestep = Timestep(episode_number,timestep_data["wall_time"],timestep_data["sim_time"],timestep_data["timestep"],
                   timestep_data["state"], timestep_data["action"], timestep_data["reward"], None)
        if len(self.current_buffer) > 1:
            self.current_buffer[-1].next_state = timestep_data["state"]
        self.current_buffer.append(timestep)

    def remove_episode(self, episode_number):
        for i in list(self.current_buffer):
            if i.episode == episode_number:
                self.current_buffer.remove(i)

    def remove_timestep(self, episode_number, timestep_number):
        for i in list(self.current_buffer):
            if i.episode == episode_number and i.timestep == timestep_number:
                self.current_buffer.remove(i)

    def get_episode(self, episode_number):
        for i in list(self.current_buffer):
            if i.episode == episode_number:
                return i

    def get_timestep(self, episode_number, timestep_number):
        for i in list(self.current_buffer):
            if i.episode == episode_number and i.timestep == timestep_number:
                return i

    def get_random_timestep_sample(self, num_timesteps):
        timestep_list = []
        if len(self.current_buffer) >= num_timesteps:
            for i in range(num_timesteps):
                timestep_list.append(self.current_buffer[randint(0,len(self.current_buffer)-1)])
        return timestep_list

    def get_recent_timestep_sample(self, num_timesteps):
        timestep_list = []
        if len(self.current_buffer) >= num_timesteps:
            for i in range(num_timesteps):
                timestep_list.append(self.current_buffer[-(i+1)])
        return timestep_list

    def get_oldest_timestep_sample(self, num_timesteps):
        timestep_list = []
        if len(self.current_buffer) >= num_timesteps:
            for i in range(num_timesteps):
                timestep_list.append(self.current_buffer[i])
        return timestep_list


if __name__ == '__main__':
    duh = ReplayBuffer(episodes_file="data/cube_all.csv")
    print(len(duh.current_buffer))
    duh.remove_episode(1)
    print(len(duh.current_buffer))
    duh.remove_timestep(1, 5)
    duh.remove_timestep(0, 5)
    print(len(duh.current_buffer))



    if duh.shuffled_buffer == duh.current_buffer:
        print("NOPE")
    
    duh.shuffled_buffer = deepcopy(duh.current_buffer)

    if duh.shuffled_buffer == duh.current_buffer:
        print("YUP")


    print(duh.get_random_timestep_sample(3))
    print("")
    print(duh.get_recent_timestep_sample(3))
    print("")
    print(duh.get_oldest_timestep_sample(3))
    print("")
