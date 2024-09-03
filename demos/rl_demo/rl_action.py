from mojograsp.simcore.action import Action
import numpy as np

class ExpertAction(Action):
    def __init__(self):
        self.current_action = {}

    def set_action(self, joint_angles: list, actor_output: list):
        self.current_action["target_joint_angles"] = joint_angles
        self.current_action['actor_output'] = actor_output

    def get_action(self) -> dict:
        # print('action from get action', self.current_action)
        return self.current_action.copy()
    
    def get_joint_angles(self):
        return self.current_action["target_joint_angles"]
    
class IKAction(Action):
    def __init__(self):
        self.current_action = {}

    def set_action(self, joint_angles: list):
        self.current_action["target_joint_angles"] = joint_angles
        

    def get_action(self) -> dict:
        # print('action from get action but the other one', self.current_action)
        return self.current_action.copy()
    
    def get_joint_angles(self):
        return self.current_action["target_joint_angles"]
    
class InterpAction(ExpertAction):
    def __init__(self, end_frequency):
        self.current_action = {}
        self.interp_ratio = int(240/end_frequency)
        self.action_phase = 0
        self.action_profile = []
        self.previous_angles = []
        self.current_action["target_joint_angles"] = [0,0,0,0]
        
    def get_joint_angles(self):
        self.action_phase +=1        
        return self.action_profile[self.action_phase-1]
    
    def set_action(self, joint_angles: list, actor_output: list):
        self.previous_angles = self.current_action["target_joint_angles"].copy()
        self.current_action["target_joint_angles"] = joint_angles
        self.current_action['actor_output'] = actor_output
        self.action_phase = 0
        self.build_action()

    def build_action(self):
        """Builds the action profile (speed profile to get from old speed to
        new speed)"""
        target = np.array(self.current_action['target_joint_angles'])
        if np.shape(target) == (self.interp_ratio,4):
            self.action_profile = target
        else:
            start = np.array(self.previous_angles)
            num_angs = len(start)
            # add_portion = target-start
            # sin_angle_thing = np.linspace(np.zeros(num_angs),np.ones(num_angs)*np.pi/2, self.interp_ratio)
            # self.action_profile = add_portion* np.sin(sin_angle_thing) + start
            self.action_profile = np.ones((self.interp_ratio,num_angs)) * target
            # self.action_profile = np.linspace(start, target,self.interp_ratio)
            # alternative using some sinusoidal stuff
        # print(f'action profile {self.action_profile}')

    def set_high_level_action(self, goal):
        self.current_action['high_level_action'] = goal