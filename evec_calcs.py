import pickle as pkl
import os
import re
import numpy as np


folder = '/home/mothra/mojo-grasp/demos/rl_demo/data/ftp_eigen_2/Test'
# print('need to load in episode all first')
print('this will be slow, and we both know it')

# get list of pkl files in folder
episode_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith('.pkl')]
filenames_only = [f for f in os.listdir(folder) if f.lower().endswith('.pkl')]

filenums = [re.findall('\d+',f) for f in filenames_only]
final_filenums = []
for i in filenums:
    if len(i) > 0 :
        final_filenums.append(int(i[0]))


sorted_inds = np.argsort(final_filenums)
final_filenums = np.array(final_filenums)
temp = final_filenums[sorted_inds]
episode_files = np.array(episode_files)
filenames_only = np.array(filenames_only)

episode_files = episode_files[sorted_inds].tolist()
# filenames_only = filenames_only[sorted_inds].tolist()
rewards = []
temp = 0
count = 0
min_eival = 100
max_eival = -100
min_eivec = 100
max_eivec = -100
for episode_file in episode_files:
    with open(episode_file, 'rb') as ef:
        tempdata = pkl.load(ef)
        eigenvalues = [i['state']['two_finger_gripper']['eigenvalues'] for i in tempdata['timestep_list']]
        eigenvectors = [i['state']['two_finger_gripper']['eigenvectors'] for i in tempdata['timestep_list']]
    a = np.min(eigenvalues)
    b = np.max(eigenvalues)
    # print(b)
    c = np.min(eigenvectors)
    d = np.max(eigenvectors)
    min_eival = min(a,min_eival)
    max_eival = max(b,max_eival)
    min_eivec = min(c,min_eivec)
    max_eivec = max(d,max_eivec)




print(min_eival, max_eival)
print(min_eivec, max_eivec)