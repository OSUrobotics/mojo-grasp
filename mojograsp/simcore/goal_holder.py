import numpy as np

class GoalHolder():
    def __init__(self, goal_pose, goal_orientation = None,goal_finger = None, goal_names = None):
        self.pose = goal_pose
        self.name = 'goal_pose'
        # print(goal_orientation)
        if goal_orientation is not None:
            self.orientation = goal_orientation
        else:
            self.orientation = np.array([None] * len(self.pose))
        
        if goal_finger is not None:
            self.finger = goal_finger
        else:
            self.finger = np.array([None,None,None,None] * len(self.pose))
        print(type(self.orientation), type(self.pose))
        self.len = len(self.pose)
        self.goal_names = goal_names
        print(f'orientation type: {type(self.orientation)}')
        print(f'lengths: pos {len(self.pose)}, orientation {len(self.orientation)}, finger {len(self.finger)}')
        if len(np.shape(self.pose)) == 1:
            self.pose = [self.pose]
        self.run_num = 0
    
    def get_data(self):
        return {'goal_position':self.pose[self.run_num%self.len],'goal_orientation':self.orientation[self.run_num%self.len], 'goal_finger':self.finger[self.run_num%self.len]}
    
    def get_name(self):
        try:
            return self.goal_names[self.run_num%self.len]
        except TypeError:
            return 'Evaluate'
    
    def next_run(self):
        self.run_num +=1
        # self.check_data()
        # print({'goal_pose':self.pose[self.run_num%self.len],'goal_orientation':self.orientation[self.run_num%self.len]})
        
    def reset(self):
        self.run_num = 0
        inds = list(range(len(self.pose)))
        np.random.shuffle(inds)
        self.pose = self.pose[inds]
        self.orientation = self.orientation[inds]
        self.finger = self.finger[inds]
        print('shuffling the pose order')
    
    def check_data(self):
        assert (self.pose[self.run_num%self.len][0] < self.finger[self.run_num%self.len][0]) and (self.pose[self.run_num%self.len][0] > self.finger[self.run_num%self.len][2])
        
    
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