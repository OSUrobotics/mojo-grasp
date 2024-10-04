import numpy as np

class GoalAverager():
    def __init__(self, num_avged):
        self.goals = []
        self.max_goals = num_avged

    def get_goal(self):
        # print('getting goal', len(self.goals))
        if len(self.goals)>1:
            # print('averaged goals',np.average(self.goals, axis=0).tolist())
            return np.average(self.goals, axis=0).tolist()
        else:
            return self.goals[0]

    def reset(self):
        self.goals = []

    def add_goal(self, new_goal):
        self.goals.append(new_goal)
        if len(self.goals) > self.max_goals:
            self.goals.pop(0)


        
class GoalHolder():
    def __init__(self, goal_pose, finger_start, goal_orientation = None,goal_finger = None, goal_names = None, mix_orientation=False, mix_finger=False):
        self.pose = goal_pose
        self.name = 'goal_pose'
        self.mix_orientation = mix_orientation
        self.mix_finger = mix_finger
        if goal_orientation is not None:
            self.orientation = goal_orientation
        else:
            self.orientation = np.array([None] * len(self.pose))
        
        if goal_finger is not None:
            self.finger = goal_finger
        else:
            self.finger = np.array([None,None,None,None] * len(self.pose))
        
        self.finger_start = finger_start
        # print(np.shape(self.finger_start))
        self.len = len(self.pose)
        self.goal_names = goal_names
        if len(np.shape(self.pose)) == 1:
            self.pose = [self.pose]
        self.run_num = 0
    
    def get_data(self):
        # print('data stuff', self.orientation)
        return {'goal_position':self.pose[self.run_num%self.len],'goal_orientation':self.orientation[self.run_num%self.len], 'goal_finger':self.finger[self.run_num%self.len]}
    
    def get_name(self):
        try:
            return self.goal_names[self.run_num%self.len]
        except TypeError:
            return 'Evaluate'
    
    def next_run(self):
        self.run_num +=1
        return [self.finger_start[0][self.run_num%self.len],self.finger_start[1][self.run_num%self.len]]

        
    def reset(self):
        self.run_num = 0
        inds = list(range(len(self.pose)))
        np.random.shuffle(inds)
        self.pose = self.pose[inds]
        if self.mix_orientation:
            np.random.shuffle(self.orientation)
        else:
            self.orientation = self.orientation[inds]
        if self.mix_finger:
            np.random.shuffle(self.finger_start)
        else:
            self.finger_start[0]=self.finger_start[0][inds]
            self.finger_start[1]=self.finger_start[1][inds]
        self.finger = self.finger[inds]
        print('shuffling the pose order')
    
    def check_data(self):
        assert (self.pose[self.run_num%self.len][0] < self.finger[self.run_num%self.len][0]) and (self.pose[self.run_num%self.len][0] > self.finger[self.run_num%self.len][2])

    def set_all_pose(self, pos, orient=None):
        '''
        This is a sneaky backdoor to allow you to change the goal position of all the datapoints to the same thing.
        Mostly useful for rotation testing
        '''    
        # print('WE SHOULDNT BE IN GERE')
        if type(self.pose) is np.ndarray:
            self.pose[:] = pos
        else:
            self.pose = np.array(self.pose)
            self.pose[:] = pos
            self.pose = self.pose.tolist()
        if orient is not None:
            if type(self.orientation) is np.ndarray:
                self.orientation[:] = orient
            else:
                self.orientation = np.array(self.orientation)
                self.orientation[:] = orient
                self.orientation.tolist()
    def __len__(self):
        return len(self.pose)
    
class RandomGoalHolder(GoalHolder):
    def __init__(self, radius_range: list):
        self.name = 'goal_pose'
        self.rrange = radius_range
        self.pose = []
        self.next_run()
    
    def next_run(self):
        l = np.sqrt(np.random.uniform(self.rrange[0]**2,self.rrange[1]**2))
        ang = np.pi * np.random.uniform(0,2)
        self.pose = [l * np.cos(ang),l * np.sin(ang)]
    
    def get_data(self):
        return {'goal_position':self.pose}
    
class SimpleGoalHolder(GoalHolder):
    def __init__(self, direction):
        self.direction = direction
        self.name = 'direction'
    def next_run(self):
        pass

    def get_data(self):
        return {'goal_direction': self.direction}
    
    def __len__(self):
        return 1
    
    def check_data(self):
        pass

    def get_name(self):
        return None
    
    def reset(self):
        pass

class SingleGoalHolder(GoalHolder):
    def __init__(self, pose):
        self.pose = pose
        self.name = 'goal_pose'
        self.len = 1

    def next_run(self):
        pass

    def get_data(self):
        # print(self.pose)
        return {'goal_position': self.pose[0], 'goal_orientation':0.0, 'goal_finger': None}
    
    def __len__(self):
        return 1
    
    def check_data(self):
        pass

    def get_name(self):
        return None
    
    def reset(self):
        pass

    def set_goal(self,goal):
        self.pose = [goal]
        
    def next_run(self):
        return [0.0,0.0]

class HRLGoalHolder(GoalHolder):
    def __init__(self, goal_pose, finger_start, goal_orientation=None, goal_finger=None, goal_names=None, mix_orientation=False, mix_finger=False, goals_smoothed=1):
        super().__init__(goal_pose, finger_start, goal_orientation, goal_finger, goal_names, mix_orientation, mix_finger)
        self.lower_pos=GoalAverager(goals_smoothed)
        self.lower_or=GoalAverager(goals_smoothed)
        self.lower_finger=GoalAverager(goals_smoothed)
        self.lower_finger.add_goal([0,0.06])
        self.lower_or.add_goal(0)
        self.lower_pos.add_goal([0,0])

    def set_pose(self, pos, orient=0, finger_separation=[0,0]):
        '''
        This is used by the higher level policy to set the goals for the lower level policy
        '''
        # print('setting pose')
        self.lower_pos.add_goal(pos)
        self.lower_or.add_goal(orient)
        self.lower_finger.add_goal(finger_separation)
        # print(self.lower_pos.get_goal(), self.lower_or.get_goal(), self.lower_finger.get_goal())

    def reset(self):
        super().reset()
        self.lower_finger.reset()
        self.lower_or.reset()
        self.lower_pos.reset()
        self.lower_finger.add_goal([0,0.06])
        self.lower_or.add_goal(0)
        self.lower_pos.add_goal([0,0])
    
    def next_run(self):
        temp  = super().next_run()
        self.lower_finger.reset()
        self.lower_or.reset()
        self.lower_pos.reset()
        self.lower_finger.add_goal([0,0.06])
        self.lower_or.add_goal(0)
        self.lower_pos.add_goal([0,0])
        return temp
    
    def get_data(self):
        return {'goal_position':self.lower_pos.get_goal(),'goal_orientation':self.lower_or.get_goal(), 'goal_finger':self.lower_finger.get_goal(),
                'upper_goal_position':self.pose[self.run_num%self.len],'upper_goal_orientation':self.orientation[self.run_num%self.len], 'upper_goal_finger':self.finger[self.run_num%self.len]}

class HRLMultigoalHolder(HRLGoalHolder):
    def __init__(self, goal_pose, finger_start,mix_orientation=False, mix_finger=False, goals_smoothed=1, num_goals_present=5, radius=0.01):
        super().__init__(goal_pose, finger_start, None, None, None, mix_orientation, mix_finger, goals_smoothed)
        self.upper_position_num = 0
        self.upper_position_set = np.array(self.pose[self.upper_position_num:self.upper_position_num+num_goals_present])
        self.num_goals_present = 5
        self.radius = radius
        self.tsteps_left = 25
        self.goals_reached = 0
        # print('making a multigoal holder')

    def get_data(self):
        # print('output of goal position', self.upper_position_set.flatten().tolist())
        return {'goal_position':self.lower_pos.get_goal(),'goal_orientation':self.lower_or.get_goal(), 'goal_finger':self.lower_finger.get_goal(),
                'upper_goal_position':self.upper_position_set.flatten().tolist(),'upper_goal_orientation':self.orientation[self.run_num%self.len], 'upper_goal_finger':self.finger[self.run_num%self.len],
                'timesteps_remaining':self.tsteps_left, 'goals_reached': self.goals_reached}
    
    def check_goal(self, object_pos):
        dists = np.linalg.norm(self.upper_position_set-object_pos, axis=1)
        self.goals_reached = 0
        self.tsteps_left -= 1
        for i, distance in enumerate(dists):
            if distance < self.radius:
                self.upper_position_num+=1
                if self.upper_position_num + self.num_goals_present>=self.len:
                    self.upper_position_num=0
                    np.random.shuffle(self.pose)
                self.upper_position_set[i] = self.pose[self.upper_position_num+self.num_goals_present]
                self.goals_reached +=1
                self.tsteps_left += 5
                # print('dingdingding')
    
    def next_run(self):
        self.run_num +=1
        self.upper_position_num += self.num_goals_present
        if self.upper_position_num+self.num_goals_present > self.len:
            self.upper_position_num=0
            np.random.shuffle(self.pose)
        self.upper_position_set = np.array(self.pose[self.upper_position_num:self.upper_position_num+self.num_goals_present])
        self.lower_finger.reset()
        self.lower_or.reset()
        self.lower_pos.reset()
        self.lower_finger.add_goal([0,0.06])
        self.lower_or.add_goal(0)
        self.lower_pos.add_goal([0,0])
        self.tsteps_left = 25
        return [self.finger_start[0][self.run_num%self.len],self.finger_start[1][self.run_num%self.len]]
    
    def reset(self):
        super().reset()
        self.tsteps_left = 25
        self.goals_reached = 0


    