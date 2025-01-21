import pickle as pkl
from scipy.spatial.transform import Rotation as R
import os

highest = ['N_mothra_slide_rerun', 'J_HPC_rerun','N_HPC_slide_rerun']

others = ['JA_S1', 'JA_S2', 'JA_S3', 'FTP_S1', 'FTP_S2', 'FTP_S3']

inside = ['Real_A', 'Real_B']

second_side = ['RA', 'RB']

base_path = '/home/mothra/mojo-grasp/demos/rl_demo/data'

ssd_path = '/media/mothra/Samsung_T5/JUL_11_Rotation_Real_World_Tests'
def shenanigans(filename,save_file):
    print(filename)
    with open(filename, 'rb') as file:
        data = pkl.load(file)
    # print(data['timestep_list'][0]['state'])
    temp = [R.from_quat(d['state']['obj_2']['pose'][1]) for d in data['timestep_list']]
    angles = [i.as_euler('xyz').tolist() for i in temp]
    t2 = [d['state']['goal_pose']['goal_orientation'] for d in data['timestep_list']]
    # print(angles)
    for i in range(len(angles)):
        data['timestep_list'][i]['reward']['object_orientation'] = angles[i]
        data['timestep_list'][i]['reward']['goal_orientation'] = t2[i]
    # print(data)
    with open(save_file, 'wb') as file:
        pkl.dump(data,file)

# files = os.listdir('/media/mothra/Samsung_T5/JUL_11_Rotation_Real_World_Tests/')
# for file in files:
#     if '2v2' in file:
#         pass
#     elif 'state' in file:
#         pass
#     elif 'actor' in file:
#         pass
#     else:
#         temp = os.path.join('/media/mothra/Samsung_T5/FTP_S1_Mothra_Rotation_Jul_9/', file)
#         t2 = os.path.join('/home/mothra/mojo-grasp/demos/rl_demo/data/Mothra_Rotation/FTP_S1/reduced_test', file)
#         shenanigans(temp, t2)

for h in highest:
    for o in others:
        for i,j in zip(inside,second_side):
            folder = '/'.join([base_path, h, o, i])
            save_folder = '/'.join([base_path, h, o, j])
            files = os.listdir(folder)
            for file in files:
                if '2v2' in file:
                    pass
                elif 'state' in file:
                    pass
                elif 'actor' in file:
                    pass
                else:
                    temp = os.path.join(folder, file)
                    t2 = os.path.join(save_folder, file)
                    shenanigans(temp, t2)


