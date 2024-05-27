from mojograsp.simcore import data_gui_backend

import pandas as pd


sub_names =['FTP_S1','FTP_S2','FTP_S3','JA_S1','JA_S2','JA_S3']
top_names = ['Mothra_Rotation','HPC_Rotation_25','Jeremiah_Rotation','Mothra_Slide','HPC_Slide','Misc_Slide']

csv_dict = {'Action Space':[], 'State Space':[], 'Upper Folder':[], 'A Distance Mean':[], 'A Distance Std':[],
 'A Angle Mean':[],'A Angle Std':[],'A Success Rate':[],'B Distance Mean':[], 'B Distance Std':[],
 'B Angle Mean':[],'B Angle Std':[],'B Success Rate':[]}

backend = data_gui_backend.PlotBackend('./data/Mothra_Rotation/FTP_S1')
for uname in top_names:
    for lname in sub_names:
        A_path = '/'.join(['./data',uname,lname,'Eval_A'])
        B_path = '/'.join(['./data',uname,lname,'Eval_B'])
        if 'Rotation' in uname:
            r_thold = 5
            t_thold = 26
        else:
            r_thold = 50
            t_thold = 10
        keys = lname.split('_')
        
        csv_dict['Action Space'].append(keys[0])
        csv_dict['State Space'].append(keys[1])
        csv_dict['Upper Folder'].append(uname)


        [mean, std] = backend.draw_scatter_end_magic(A_path)
        csv_dict['A Distance Mean'].append(mean)
        csv_dict['A Distance Std'].append(std)
        [mean, std] = backend.draw_orientation_end_magic(A_path)
        csv_dict['A Angle Mean'].append(mean)
        csv_dict['A Angle Std'].append(std)
        sr = backend.draw_success_rate(A_path,t_thold,r_thold)
        csv_dict['A Success Rate'].append(sr)
        backend.reset()

        [mean, std] = backend.draw_scatter_end_magic(B_path)
        csv_dict['B Distance Mean'].append(mean)
        csv_dict['B Distance Std'].append(std)
        [mean, std] = backend.draw_orientation_end_magic(B_path)
        csv_dict['B Angle Mean'].append(mean)
        csv_dict['B Angle Std'].append(std)
        sr = backend.draw_success_rate(B_path,t_thold,r_thold)
        csv_dict['B Success Rate'].append(sr)
        backend.reset()

df = pd.DataFrame(csv_dict)
df.to_csv("./simulation_test_data.csv", index=False)