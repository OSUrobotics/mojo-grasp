import numpy as np


class StartHolder():
    def __init__(self, object_pos, object_orientation=None, finger_angles=None, finger_ys=None):
        self.obj_pos = object_pos
        self.len = len(object_pos)
        if object_orientation is None:
            self.obj_orientation = np.zeros(self.len)
        else:
            self.obj_orientation = object_orientation
        self.run_num = 0
        self.finger_angles = finger_angles
        self.finger_ys = finger_ys

    def next_run(self):
        self.run_num += 1

    def get_data(self):
        obj_dict = {'translation':self.obj_pos[self.run_num%self.len],'rotation':self.obj_orientation[self.run_num%self.len]}
        if self.finger_angles is not None:
            finger_dict = {'joint_angles':self.finger_angles[self.run_num%self.len]}
        elif self.finger_ys is not None:
            finger_dict = {'finger_y':self.finger_ys[self.run_num%self.len]}
        else:
            finger_dict = None
        return obj_dict, finger_dict

    def reset(self, shuffle_type='full'):
        self.run_num = 0
        inds = list(range(self.len))
        np.random.shuffle(inds)
        self.obj_pos = self.obj_pos[inds]
        if shuffle_type == 'full':
            inds = list(range(self.len))
            np.random.shuffle(inds)
            self.obj_orientation = self.obj_orientation[inds]
            if self.finger_angles is not None:
                inds = list(range(self.len))
                np.random.shuffle(inds)
                self.finger_angles = self.finger_angles[inds]
            if self.finger_ys is not None:
                inds = list(range(self.len))
                np.random.shuffle(inds)
                self.finger_ys = self.finger_ys[inds]
        elif shuffle_type == 'together':
            self.obj_orientation = self.obj_orientation[inds]
            if self.finger_angles is not None:
                self.finger_angles = self.finger_angles[inds]
            if self.finger_ys is not None:
                self.finger_ys = self.finger_ys[inds]

class RandomStartHolder():
    def __init__(self, range_dictionary):
        # Range dictionary contains a vector with the min and max values for each randomization region
        # angle ranges should be in radians
        random_keys = range_dictionary.keys()
        self.obj_dict = {}
        self.finger_dict = {}
        assert ('x' not in random_keys) or ('r' not in random_keys)
        if 'x' in random_keys:
            self.position_randomization=self.xyRandom
            self.xlim = range_dictionary['x']
            if 'y' in random_keys:
                self.ylim = range_dictionary['y']
            else:
                self.ylim = np.array([0,0])
        elif 'r' in random_keys:
            self.position_randomization=self.rthetaRandom
            self.rlim = range_dictionary['r']
            if 'theta' in random_keys:
                self.thetalim = range_dictionary['theta']
            else:
                self.thetalim = np.array([0,0])
        else:
            self.position_randomization=self.xyRandom
            self.xlim = np.array([0,0])
            self.ylim = np.array([0,0])
        if 'orientation' in random_keys:
            self.orientationlim = range_dictionary['orientation']
        else:
            self.orientationlim = np.array([0,0])
        if 'fingery' in random_keys:
            self.fingerlim = range_dictionary['fingery']
        else:
            self.fingerlim = np.array([0,0])
        self.next_run()

    def xyRandom(self,size=1):
        x = np.random.uniform(self.xlim[0], self.xlim[1],size=size)
        y = np.random.uniform(self.ylim[0], self.ylim[1],size=size)
        return x[0],y[0]
    
    def rthetaRandom(self,size=1):
        r = np.random.uniform(0,1,size=size)
        theta = np.random.uniform(self.thetalim[0],self.thetalim[1],size=size)
        x = ((1-r**2) * (self.rlim[1]-self.rlim[0]) + self.rlim[0]) * np.sin(theta)
        y = ((1-r**2) * (self.rlim[1]-self.rlim[0]) + self.rlim[0]) * np.cos(theta)
        return x[0],y[0]
    
    def reset(self):
        pass

    def next_run(self):
        x,y = self.position_randomization()
        fingerys = np.random.uniform(self.fingerlim[0], self.fingerlim[1], 2)
        orientation = np.random.uniform(self.orientationlim[0], self.orientationlim[1])
        self.obj_dict = {'translation':[x,y], 'rotation':orientation}
        self.finger_dict = {'finger_y':fingerys}

    def get_data(self):
        print('getting data from the start dict', self.obj_dict, self.finger_dict)
        return self.obj_dict, self.finger_dict