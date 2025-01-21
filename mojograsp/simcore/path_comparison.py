import numpy as np
import json
import matplotlib.pyplot as plt
#import SiMBols as sb
import pickle as pkl

class Evaluator():
    def __init__(self, data_folder: str):
        self.folder = data_folder
        self.paths = []
        self.data_dict = {}
        
    def load_single(self, episode_num: int):
        with open(self.folder + '/episode_'+ str(episode_num)+'.json') as file:
            self.data_dict = json.load(file)
    
    def plot_path(self):
        data = self.data_dict['timestep_list']
        trajectory_points = [f['state']['obj_2']['pose'][0] for f in data]
        goal_pose = data[0]['reward']['goal_position']
        trajectory_points = np.array(trajectory_points)
        plt.plot(trajectory_points[:,0], trajectory_points[:,1])
        plt.plot([trajectory_points[0,0], goal_pose[0]],[trajectory_points[0,1],goal_pose[1]])
        plt.xlim([-0.07,0.07])
        plt.ylim([0.1,0.22])
        print(goal_pose)
        plt.show()
#    def evaluate_single(self, episode_num: int):
#        with open(self.folder + '/episode_' + str(episode_num)+'.json') as file:
#            data_dict = json.load(file)
#        data = data_dict['timestep_list']
#        trajectory_points = [f['state']['obj_2']['pose'][0] for f in data]
#        goal_pose = data[0]['reward']['goal_position']
#        goal_pose[2] = 0.05
#        start_pose = trajectory_points[0]
#
#        full_trajectory = np.array(trajectory_points)
#        real_trajectory = sb.Trajectory(full_trajectory)
#        ideal_trajectory = np.linspace(start_pose,goal_pose,num=len(trajectory_points))
#        ideal_trajectory = sb.Trajectory(ideal_trajectory)
#        tester = sb.Comparer(real_trajectory,ideal_trajectory)
#        tester.dfd()
#        print(tester.DFD)
#        rewards = [f['reward']['distance_to_goal'] for f in data]
#        print('closest point to the goal', min(rewards))


    def plot_angles(self):
        data = self.data_dict['timestep_list']
        current_angle_dict = [f['state']['two_finger_gripper']['joint_angles'] for f in data]
        current_angle_list = []
        for angle in current_angle_dict:
            temp = [angs for angs in angle.values()]
            current_angle_list.append(temp)        
        current_action_list= [f['action']['target_joint_angles'] for f in data]
        
        current_angle_list=np.array(current_angle_list)
        current_action_list=np.array(current_action_list)
        goal_pose = data[5]['reward']['goal_position']
        print(goal_pose)
        angle_tweaks = current_angle_list#current_action_list - current_angle_list
        plt.plot(range(len(angle_tweaks)),angle_tweaks[:,0])
        plt.plot(range(len(angle_tweaks)),angle_tweaks[:,1])
        plt.plot(range(len(angle_tweaks)),angle_tweaks[:,2])
        plt.plot(range(len(angle_tweaks)),angle_tweaks[:,3])
        plt.legend(['Angle 1', 'Angle 2', 'Angle 3', 'Angle 4'])
        plt.ylabel('Angle (radians)')
        plt.xlabel('Timestep (1/240 s)')
        plt.title('Middle Right Action')
        plt.show()

    def evaluate_group(self, num_eps: list):
        for num in num_eps:
            with open(self.folder + 'episode_' + str(num)+'.json') as file:
                data_dict = json.load(file)

class EvaluatorPKL(Evaluator):
    def __init__(self, data_folder: str):
        super().__init__(data_folder)
        self.data_dict = {}

    def load_single(self, episode_num: int):
        with open(self.folder + '/episode_'+ str(episode_num)+'.pkl', 'rb') as pkl_file:
            self.data_dict = pkl.load(pkl_file)

#    def evaluate_single(self):
#        data = self.data_dict['timestep_list']
#        trajectory_points = [f['state']['obj_2']['pose'][0] for f in data]
#        goal_pose = data[0]['reward']['goal_position']
#        goal_pose[2] = 0.05
#        start_pose = trajectory_points[0]
#
#        full_trajectory = np.array(trajectory_points)
#        real_trajectory = sb.Trajectory(full_trajectory)
#        ideal_trajectory = np.linspace(start_pose,goal_pose,num=len(trajectory_points))
#        ideal_trajectory = sb.Trajectory(ideal_trajectory)
#        tester = sb.Comparer(real_trajectory,ideal_trajectory)
#        tester.dfd()
#        print(tester.DFD)
#        rewards = [f['reward']['distance_to_goal'] for f in data]
#        print('closest point to the goal', min(rewards))

    def plot_path(self):
        data = self.data_dict['timestep_list']
        trajectory_points = [f['state']['obj_2']['pose'][0] for f in data]
        goal_pose = data[5]['reward']['goal_position']
        print(goal_pose)
        trajectory_points = np.array(trajectory_points)
        plt.plot(trajectory_points[:,0], trajectory_points[:,1])
        plt.plot([trajectory_points[0,0], goal_pose[0]],[trajectory_points[0,1],goal_pose[1]])
        plt.xlim([-0.07,0.07])
        plt.ylim([0.1,0.22])
        plt.ylabel('X pos (m)')
        plt.xlabel('Y pos (m)')
        plt.legend(['RL Trajectory', 'Ideal Path to Goal'])
        plt.title('Late Left Action')
        plt.show()
        
    
        
try:
    test = EvaluatorPKL('./left')
    test.load_single(984)
#    test.plot_path()
    test.plot_angles()
    
except FileNotFoundError:
    test = Evaluator('./left')
    test.load_single(984)
#    test.plot_path()
    test.plot_angles()
    
    
    