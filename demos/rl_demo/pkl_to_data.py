import os
import re
import numpy as np
import pandas as pd
import multiprocessing
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import pickle as pkl

def slim_pool_process(episode_file):
    with open(episode_file, 'rb') as ef:
        tempdata = pkl.load(ef)

    data = tempdata['timestep_list']
    point_list = []

    for timestep in data: 
        row = [
            timestep['state']['obj_2']['pose'][0][0], 
            timestep['state']['obj_2']['pose'][0][1],  
            timestep['state']['obj_2']['pose'][0][2],  
            *timestep['state']['obj_2']['pose'][1][0:4]  
        ]
        point_list.append(row)

    return point_list

class DataProcessor:
    def __init__(self):
        self.point_dictionary = None
        self.fig, self.ax = plt.subplots()

    def build_slim(self, folder_path):
        if not isinstance(folder_path, str):
            raise ValueError('folder path must be a string')

        episode_files = [
            os.path.join(folder_path, f) for f in os.listdir(folder_path)
            if f.lower().endswith('.pkl') and '2v2' not in f
        ]
        filenames_only = [os.path.basename(f) for f in episode_files]
        filenums = [re.findall(r'\d+', f) for f in filenames_only]
        final_filenums = [int(i[0]) for i in filenums if i]
        sorted_inds = np.argsort(final_filenums)
        episode_files = np.array(episode_files)[sorted_inds].tolist()

        with multiprocessing.Pool() as pool:
            data_list = pool.map(slim_pool_process, episode_files)

        flattened_data = [row for episode in data_list for row in episode]
        column_key = ['X', 'Y', 'Z', 'x_q', 'y_q', 'z_q', 'w_q']

        self.point_dictionary = pd.DataFrame(flattened_data, columns=column_key)

    def clear_axes(self):
        self.ax.clear()

    def draw_XY(self, folder_path):
        if self.point_dictionary is None:
            self.build_slim(folder_path)
        
        self.clear_axes()
        x = self.point_dictionary['X']
        y = self.point_dictionary['Y']
        
        self.ax.scatter(x, y, s=2)
        self.ax.set_xlabel('X position (cm)')
        self.ax.set_ylabel('Y position (cm)')
        self.ax.set_title('X-Y Map')
        plt.show()

    def draw_Q_bins(self, folder_path):
        if self.point_dictionary is None:
            self.build_slim(folder_path)

        self.clear_axes()
        qx = self.point_dictionary['x_q']
        qy = self.point_dictionary['y_q']
        qz = self.point_dictionary['z_q']
        qw = self.point_dictionary['w_q']

        euler_angles = np.array([R.from_quat([qx[i], qy[i], qz[i], qw[i]]).as_euler('xyz', degrees=True) for i in range(len(qx))])
        pitch = euler_angles[:, 0]
        roll = euler_angles[:, 1]
        yaw = euler_angles[:, 2]

        pitch_bins = np.linspace(-180, 180, 40)
        roll_bins = np.linspace(-180, 180, 40)
        yaw_bins = np.linspace(-180, 180, 40)

        pitch_hist, _ = np.histogram(pitch, bins=pitch_bins)
        roll_hist, _ = np.histogram(roll, bins=roll_bins)
        yaw_hist, _ = np.histogram(yaw, bins=yaw_bins)

        self.ax.hist(pitch_bins[:-1], bins=pitch_bins, weights=pitch_hist, alpha=0.5, label='Pitch')
        self.ax.hist(roll_bins[:-1], bins=roll_bins, weights=roll_hist, alpha=0.5, label='Roll')
        self.ax.hist(yaw_bins[:-1], bins=yaw_bins, weights=yaw_hist, alpha=0.5, label='Yaw')
        self.ax.set_xlabel('Angle (degrees)')
        self.ax.set_ylabel('Frequency')
        self.ax.set_title('Orientation Bins')
        self.ax.legend()
        plt.show()

def main():
    processor = DataProcessor()
    folder_path = input("Enter the folder path containing .pkl files: ")
    
    while True:
        print("\nChoose an option:")
        print("1. Build dataset")
        print("2. Draw XY scatter plot")
        print("3. Draw orientation bins")
        print("4. Exit")
        choice = input("Enter your choice: ")

        if choice == '1':
            processor.build_slim(folder_path)
            print("Dataset built successfully.")
        elif choice == '2':
            processor.draw_XY(folder_path)
        elif choice == '3':
            processor.draw_Q_bins(folder_path)
        elif choice == '4':
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
