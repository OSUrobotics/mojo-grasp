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
        goal_dict = {'translation':self.obj_pos[self.run_num%self.len],'rotation':self.obj_orientation[self.run_num%self.len]}
        if self.finger_angles is not None:
            finger_dict = {'joint_angles':self.finger_angles[self.run_num%self.len]}
        elif self.finger_ys is not None:
            finger_dict = {'finger_y':self.finger_ys[self.run_num%self.len]}
        else:
            finger_dict = None
        return goal_dict, finger_dict

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

