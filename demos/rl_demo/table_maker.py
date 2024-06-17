from mojograsp.simcore import data_gui_backend

import pandas as pd

'''
sub_names =['FTP_S1','FTP_S2','FTP_S3','JA_S1','JA_S2','JA_S3']
top_names = ['Mothra_Slide','HPC_Slide','Misc_Slide','Mothra_Rotation','HPC_Rotation','Jeremiah_Rotation','Mothra_Full','HPC_Full','Jeremiah_Full']

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
            r_thold = 900
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
df.to_csv("./simulation_test_data_3rd_round.csv", index=False)


# Begin real world comparison table making
sub_names =['FTP_S1','FTP_S2','FTP_S3','JA_S1','JA_S2','JA_S3']
top_names_slide = ['Mothra_Slide','HPC_Slide','Misc_Slide']
top_names_rotate = ['Mothra_Rotation','HPC_Rotation','Jeremiah_Rotation']
top_names_full = ['Mothra_Full','HPC_Full','Jeremiah_Full']

csv_dict = {'Action Space':[], 'State Space':[], 'Sim A Distance Covered':[], 'Sim A Distance Std':[],
 'Sim A Efficiency':[],'Sim A Efficiency Std':[],'Real A Distance Covered':[], 'Real A Distance Std':[],
 'Real A Efficiency':[],'Real A Efficiency Std':[], 'Sim B Distance Covered':[], 'Sim B Distance Std':[],
 'Sim B Efficiency':[],'Sim B Efficiency Std':[],'Real B Distance Covered':[], 'Real B Distance Std':[],
 'Real B Efficiency':[],'Real B Efficiency Std':[]}

backend = data_gui_backend.PlotBackend('./data/Mothra_Rotation/FTP_S1')
for lname in sub_names:
    A_path = ['/'.join(['./data',uname,lname,'Ast_A']) for uname in top_names_slide]
    B_path = ['/'.join(['./data',uname,lname,'Ast_B']) for uname in top_names_slide]
    Real_A_path = ['/'.join(['./data',uname,lname,'Real_A']) for uname in top_names_slide]
    Real_B_path = ['/'.join(['./data',uname,lname,'Real_B']) for uname in top_names_slide]

    keys = lname.split('_')
    
    csv_dict['Action Space'].append(keys[0])
    csv_dict['State Space'].append(keys[1])
    # csv_dict['Upper Folder'].append(uname)


    asterisk_radar= backend.draw_radar(A_path,'No')
    csv_dict['Sim A Distance Covered'].append(asterisk_radar[0])
    csv_dict['Sim A Distance Std'].append(asterisk_radar[1])
    csv_dict['Sim A Efficiency'].append(asterisk_radar[2])
    csv_dict['Sim A Efficiency Std'].append(asterisk_radar[3])

    asterisk_radar= backend.draw_radar(B_path,'No')
    csv_dict['Sim B Distance Covered'].append(asterisk_radar[0])
    csv_dict['Sim B Distance Std'].append(asterisk_radar[1])
    csv_dict['Sim B Efficiency'].append(asterisk_radar[2])
    csv_dict['Sim B Efficiency Std'].append(asterisk_radar[3])

    asterisk_radar= backend.draw_radar(Real_A_path,'No')
    csv_dict['Real A Distance Covered'].append(asterisk_radar[0])
    csv_dict['Real A Distance Std'].append(asterisk_radar[1])
    csv_dict['Real A Efficiency'].append(asterisk_radar[2])
    csv_dict['Real A Efficiency Std'].append(asterisk_radar[3])

    asterisk_radar= backend.draw_radar(Real_B_path,'No')
    csv_dict['Real B Distance Covered'].append(asterisk_radar[0])
    csv_dict['Real B Distance Std'].append(asterisk_radar[1])
    csv_dict['Real B Efficiency'].append(asterisk_radar[2])
    csv_dict['Real B Efficiency Std'].append(asterisk_radar[3])
    backend.reset()

df = pd.DataFrame(csv_dict)
df.to_csv("./sim_real_comparison_slide.csv", index=False)
'''
csv_dict = {'Action Space':[], 'State Space':[], 'Sim A Distance Error':[], 'Sim A Distance Std':[],
 'Sim A Orientation Error':[],'Sim A Orientation Std':[], 'Sim A Orientation Traveled':[], 'Sim A Traveled Std':[],'Real A Distance Error':[], 'Real A Distance Std':[],
 'Real A Orientation Error':[],'Real A Orientation Std':[], 'Real A Orientation Traveled':[], 'Real A Traveled Std':[],'Sim B Distance Error':[], 'Sim B Distance Std':[],
 'Sim B Orientation Error':[],'Sim B Orientation Std':[], 'Sim B Orientation Traveled':[], 'Sim B Traveled Std':[],'Real B Distance Error':[], 'Real B Distance Std':[],
 'Real B Orientation Error':[],'Real B Orientation Std':[], 'Real B Orientation Traveled':[], 'Real B Traveled Std':[]}



# Begin real world comparison table making
sub_names =['FTP_S1','FTP_S2','FTP_S3','JA_S1','JA_S2','JA_S3']
top_names_slide = ['Mothra_Slide','HPC_Slide','Misc_Slide']
top_names_rotate = ['Mothra_Rotation','HPC_Rotation','Jeremiah_Rotation']
top_names_full = ['Mothra_Full','HPC_Full','Jeremiah_Full']

backend = data_gui_backend.PlotBackend('./data/Mothra_Rotation/FTP_S1')
for lname in sub_names:
    A_path = ['/'.join(['./data',uname,lname,'Ast_A']) for uname in top_names_rotate]
    B_path = ['/'.join(['./data',uname,lname,'Ast_B']) for uname in top_names_rotate]
    Real_A_path = ['/'.join(['./data',uname,lname,'Real_A']) for uname in top_names_rotate]
    Real_B_path = ['/'.join(['./data',uname,lname,'Real_B']) for uname in top_names_rotate]

    keys = lname.split('_')
    
    csv_dict['Action Space'].append(keys[0])
    csv_dict['State Space'].append(keys[1])

    try:
        asterisk_radar= backend.draw_rotation_asterisk(A_path,'Sim A')
        csv_dict['Sim A Distance Error'].append(asterisk_radar[0])
        csv_dict['Sim A Distance Std'].append(asterisk_radar[1])
        csv_dict['Sim A Orientation Error'].append(asterisk_radar[2])
        csv_dict['Sim A Orientation Std'].append(asterisk_radar[3])
        csv_dict['Sim A Orientation Traveled'].append(asterisk_radar[4])
        csv_dict['Sim A Traveled Std'].append(asterisk_radar[5])
    except:
        print(f'error in Sim A, subname: {lname}, continuing crunching')
    try:
        asterisk_radar= backend.draw_rotation_asterisk(B_path,'Sim B')
        csv_dict['Sim B Distance Error'].append(asterisk_radar[0])
        csv_dict['Sim B Distance Std'].append(asterisk_radar[1])
        csv_dict['Sim B Orientation Error'].append(asterisk_radar[2])
        csv_dict['Sim B Orientation Std'].append(asterisk_radar[3])
        csv_dict['Sim B Orientation Traveled'].append(asterisk_radar[4])
        csv_dict['Sim B Traveled Std'].append(asterisk_radar[5])
    except:
        print(f'error in Sim B, subname: {lname}, continuing crunching')
    try:
        asterisk_radar= backend.draw_rotation_asterisk(Real_A_path,'Real A')
        csv_dict['Real A Distance Error'].append(asterisk_radar[0])
        csv_dict['Real A Distance Std'].append(asterisk_radar[1])
        csv_dict['Real A Orientation Error'].append(asterisk_radar[2])
        csv_dict['Real A Orientation Std'].append(asterisk_radar[3])
        csv_dict['Real A Orientation Traveled'].append(asterisk_radar[4])
        csv_dict['Real A Traveled Std'].append(asterisk_radar[5])
    except:
        print(f'error in Real A, subname: {lname}, continuing crunching')
    try:
        asterisk_radar= backend.draw_rotation_asterisk(Real_B_path,'Real B')
        csv_dict['Real B Distance Error'].append(asterisk_radar[0])
        csv_dict['Real B Distance Std'].append(asterisk_radar[1])
        csv_dict['Real B Orientation Error'].append(asterisk_radar[2])
        csv_dict['Real B Orientation Std'].append(asterisk_radar[3])
        csv_dict['Real B Orientation Traveled'].append(asterisk_radar[4])
        csv_dict['Real B Traveled Std'].append(asterisk_radar[5])
    except:
        print(f'error in Real B, subname: {lname}, continuing crunching')
    backend.reset()

df = pd.DataFrame(csv_dict)
df.to_csv("./sim_real_comparison_rotate.csv", index=False)
