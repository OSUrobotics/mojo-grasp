from mojograsp.simcore import data_gui_backend

import pandas as pd


sub_names =['FTP_S1','FTP_S2','FTP_S3','JA_S1','JA_S2','JA_S3']
top_names =['Mothra_Rotation','HPC_Rotation','Jeremiah_Rotation'] #['Sliding_B']  #['Mothra_Rotation','HPC_Rotation','Jeremiah_Rotation','Mothra_Full','HPC_Full','Jeremiah_Full']

csv_dict = {'Action Space':[], 'State Space':[], 'Upper Folder':[], 'Sim A Distance Mean':[], 'Sim A Distance Std':[],
 'Sim A Angle Mean':[],'Sim A Angle Std':[],'Sim A Success Rate':[],'Sim B Distance Mean':[], 'Sim B Distance Std':[],
 'Sim B Angle Mean':[],'Sim B Angle Std':[],'Sim B Success Rate':[], 'Real A Distance Mean':[], 'Real A Distance Std':[],
 'Real A Angle Mean':[],'Real A Angle Std':[],'Real A Success Rate':[],'Real B Distance Mean':[], 'Real B Distance Std':[],
 'Real B Angle Mean':[],'Real B Angle Std':[],'Real B Success Rate':[]}

backend = data_gui_backend.PlotBackend()
'''
for uname in top_names:
    for lname in sub_names:
        SimApath = '/'.join(['./data',uname,lname,'Ast_A'])
        SimBpath = '/'.join(['./data',uname,lname,'Ast_B'])
        RealApath= '/'.join(['./data',uname,lname,'RA3'])
        RealBpath = '/'.join(['./data',uname,lname,'RB3'])
        if 'Rotation' in uname:
            r_thold = 5
            t_thold = 26
        elif 'Full' in uname:
            r_thold = 10
            t_thold = 10
        else:
            r_thold = 900
            t_thold = 10
        keys = lname.split('_')
        
        csv_dict['Action Space'].append(keys[0])
        csv_dict['State Space'].append(keys[1])
        csv_dict['Upper Folder'].append(uname)


        [mean, std] = backend.draw_scatter_end_magic(SimApath)
        csv_dict['Sim A Distance Mean'].append(mean)
        csv_dict['Sim A Distance Std'].append(std)
        [mean, std] = backend.draw_orientation_end_magic(SimApath)
        csv_dict['Sim A Angle Mean'].append(mean)
        csv_dict['Sim A Angle Std'].append(std)
        sr = backend.draw_success_rate(SimApath,t_thold,r_thold)
        csv_dict['Sim A Success Rate'].append(sr)
        backend.reset()

        [mean, std] = backend.draw_scatter_end_magic(SimBpath)
        csv_dict['Sim B Distance Mean'].append(mean)
        csv_dict['Sim B Distance Std'].append(std)
        [mean, std] = backend.draw_orientation_end_magic(SimBpath)
        csv_dict['Sim B Angle Mean'].append(mean)
        csv_dict['Sim B Angle Std'].append(std)
        sr = backend.draw_success_rate(SimBpath,t_thold,r_thold)
        csv_dict['Sim B Success Rate'].append(sr)
        backend.reset()

        [mean, std] = backend.draw_scatter_end_magic(RealApath)
        csv_dict['Real A Distance Mean'].append(mean)
        csv_dict['Real A Distance Std'].append(std)
        [mean, std] = backend.draw_orientation_end_magic(RealApath)
        csv_dict['Real A Angle Mean'].append(mean)
        csv_dict['Real A Angle Std'].append(std)
        sr = backend.draw_success_rate(RealApath,t_thold,r_thold)
        csv_dict['Real A Success Rate'].append(sr)
        backend.reset()

        [mean, std] = backend.draw_scatter_end_magic(RealBpath)
        csv_dict['Real B Distance Mean'].append(mean)
        csv_dict['Real B Distance Std'].append(std)
        [mean, std] = backend.draw_orientation_end_magic(RealBpath)
        csv_dict['Real B Angle Mean'].append(mean)
        csv_dict['Real B Angle Std'].append(std)
        sr = backend.draw_success_rate(RealBpath,t_thold,r_thold)
        csv_dict['Real B Success Rate'].append(sr)
        backend.reset()

df = pd.DataFrame(csv_dict)
df.to_csv("./rotation_real_World_Results.csv", index=False)

'''
# Begin real world comparison table making
sub_names =['FTP_S1','FTP_S2','FTP_S3','JA_S1','JA_S2','JA_S3']

top_names_slide = ['mslide','HPC_Slide','jslide']
top_names_rotate = ['Mothra_Rotation','HPC_Rotation','Jeremiah_Rotation']
top_names_full = ['Mothra_Full','HPC_Full','Jeremiah_Full']

other = ['Ast_A','Ast_B','Real_A','Real_B']

# csv_dict = {'Action Space':[], 'State Space':[], 'Sim A Distance Covered':[], 'Sim A Distance Std':[],
#  'Sim A Efficiency':[],'Sim A Efficiency Std':[],'Real A Distance Covered':[], 'Real A Distance Std':[],
#  'Real A Efficiency':[],'Real A Efficiency Std':[], 'Sim B Distance Covered':[], 'Sim B Distance Std':[],
#  'Sim B Efficiency':[],'Sim B Efficiency Std':[],'Real B Distance Covered':[], 'Real B Distance Std':[],
#  'Real B Efficiency':[],'Real B Efficiency Std':[]}


csv_dict = {'Action Space':[], 'State Space':[], 'Sim A Distance Covered':[], 'Sim A Distance Std':[],
 'Sim A Efficiency':[],'Sim A Efficiency Std':[],'Sim B Distance Covered':[], 'Sim B Distance Std':[],
 'Sim B Efficiency':[],'Sim B Efficiency Std':[]}

backend = data_gui_backend.PlotBackend()

# for lname in sub_names:
#     A_path = ['/'.join(['./data',uname,lname,'Ast_A']) for uname in top_names_slide]
#     B_path = ['/'.join(['./data',uname,lname,'Ast_B']) for uname in top_names_slide]
#     Real_A_path = ['/'.join(['./data',uname,lname,'Real_A']) for uname in top_names_slide]
#     Real_B_path = ['/'.join(['./data',uname,lname,'Real_B']) for uname in top_names_slide]

#     keys = lname.split('_')
    
#     csv_dict['Action Space'].append(keys[0])
#     csv_dict['State Space'].append(keys[1])
    # csv_dict['Upper Folder'].append(uname)
    # print(A_path)

    # asterisk_radar= backend.draw_radar(A_path,'Sim A')
    # csv_dict['Sim A Distance Covered'].append(asterisk_radar[0])
    # csv_dict['Sim A Distance Std'].append(asterisk_radar[1])
    # csv_dict['Sim A Efficiency'].append(asterisk_radar[2])
    # csv_dict['Sim A Efficiency Std'].append(asterisk_radar[3])

    # asterisk_radar= backend.draw_radar(B_path,'Sim B')
    # csv_dict['Sim B Distance Covered'].append(asterisk_radar[0])
    # csv_dict['Sim B Distance Std'].append(asterisk_radar[1])
    # csv_dict['Sim B Efficiency'].append(asterisk_radar[2])
    # csv_dict['Sim B Efficiency Std'].append(asterisk_radar[3])

    # asterisk_radar= backend.draw_radar(Real_A_path,'Real A')
    # csv_dict['Real A Distance Covered'].append(asterisk_radar[0])
    # csv_dict['Real A Distance Std'].append(asterisk_radar[1])
    # csv_dict['Real A Efficiency'].append(asterisk_radar[2])
    # csv_dict['Real A Efficiency Std'].append(asterisk_radar[3])

    # asterisk_radar= backend.draw_radar(Real_B_path,'Real B')
    # csv_dict['Real B Distance Covered'].append(asterisk_radar[0])
    # csv_dict['Real B Distance Std'].append(asterisk_radar[1])
    # csv_dict['Real B Efficiency'].append(asterisk_radar[2])
    # csv_dict['Real B Efficiency Std'].append(asterisk_radar[3])
    # backend.fig.savefig(lname + '_holup.png',dpi=400)
    # backend.fig.savefig(lname + '_sim_a.png',dpi=400)
    # backend.reset()
    # backend.clear_axes()

# # backend = data_gui_backend.PlotBackend()
# for o in other:
#     f1 = ['/'.join(['./data',uname,'FTP_S1',o]) for uname in top_names_slide]
#     f2 = ['/'.join(['./data',uname,'FTP_S2',o]) for uname in top_names_slide]
#     f3 = ['/'.join(['./data',uname,'FTP_S3',o]) for uname in top_names_slide]
#     # Real_B_path = ['/'.join(['./data',uname,lname,'Real_B']) for uname in top_names_slide]

#     # keys = lname.split('_')
    
#     # csv_dict['Action Space'].append(keys[0])
#     # csv_dict['State Space'].append(keys[1])
#     # csv_dict['Upper Folder'].append(uname)
#     # print(A_path)
#     # asterisk_radar= backend.draw_radar(f3,'FTP_S3')
#     # asterisk_radar= backend.draw_radar(f2,'FTP_S2')
#     # asterisk_radar= backend.draw_radar(f1,'FTP_S1')
#     # csv_dict['Sim A Distance Covered'].append(asterisk_radar[0])
#     # csv_dict['Sim A Distance Std'].append(asterisk_radar[1])
#     # csv_dict['Sim A Efficiency'].append(asterisk_radar[2])
#     # csv_dict['Sim A Efficiency Std'].append(asterisk_radar[3])

    
#     # csv_dict['Sim B Distance Covered'].append(asterisk_radar[0])
#     # csv_dict['Sim B Distance Std'].append(asterisk_radar[1])
#     # csv_dict['Sim B Efficiency'].append(asterisk_radar[2])
#     # csv_dict['Sim B Efficiency Std'].append(asterisk_radar[3])

    
#     # csv_dict['Real A Distance Covered'].append(asterisk_radar[0])
#     # csv_dict['Real A Distance Std'].append(asterisk_radar[1])
#     # csv_dict['Real A Efficiency'].append(asterisk_radar[2])
#     # csv_dict['Real A Efficiency Std'].append(asterisk_radar[3])

#     backend.fig.savefig(o + '_FTP.png',dpi=400)
#     backend.reset()
#     backend.clear_axes()
# backend = data_gui_backend.PlotBackend()
f1 = ['/'.join(['./data',uname,'FTP_S1',other[0]]) for uname in top_names_slide]
f2 = ['/'.join(['./data',uname,'FTP_S2',other[0]]) for uname in top_names_slide]
f3 = ['/'.join(['./data',uname,'FTP_S3',other[0]]) for uname in top_names_slide]
j1 = ['/'.join(['./data',uname,'JA_S1',other[0]]) for uname in top_names_slide]
j2 = ['/'.join(['./data',uname,'JA_S2',other[0]]) for uname in top_names_slide]
j3 = ['/'.join(['./data',uname,'JA_S3',other[0]]) for uname in top_names_slide]

for o in other[1:]:
    cf1 = ['/'.join(['./data',uname,'FTP_S1',o]) for uname in top_names_slide]
    cf2 = ['/'.join(['./data',uname,'FTP_S2',o]) for uname in top_names_slide]
    cf3 = ['/'.join(['./data',uname,'FTP_S3',o]) for uname in top_names_slide]
    cj1 = ['/'.join(['./data',uname,'JA_S1',o]) for uname in top_names_slide]
    cj2 = ['/'.join(['./data',uname,'JA_S2',o]) for uname in top_names_slide]
    cj3 = ['/'.join(['./data',uname,'JA_S3',o]) for uname in top_names_slide]
    try:
        print('FTP S1 comparison ' + o)
        backend.radar_fuckery(f1,cf1)
    except:
        print('FTPS1 failed')
    try:
        print('FTP S2 comparison ' + o)
        backend.radar_fuckery(f2,cf2)
    except:
        print('FTPS2 failed')
    try:
        print('FTP S3 comparison ' + o)
        backend.radar_fuckery(f3,cf3)
    except:
        print('FTPS3 failed')
    try:
        print('JA S1 comparison ' + o)
        backend.radar_fuckery(j1,cj1)
    except:
        print('JAS1 failed')
    try:
        print('JA S2 comparison ' + o)
        backend.radar_fuckery(j2,cj2)
    except:
        print('JA S2 failed')
    try:
        print('JA S3 comparison ' + o)
        backend.radar_fuckery(j3,cj3)
    except:
        print('JA S3 failed')
# for o in other:
#     f1 = ['/'.join(['./data',uname,'FTP_S1',o]) for uname in top_names_slide]
#     f2 = ['/'.join(['./data',uname,'FTP_S2',o]) for uname in top_names_slide]
#     f3 = ['/'.join(['./data',uname,'FTP_S3',o]) for uname in top_names_slide]
#     # Real_B_path = ['/'.join(['./data',uname,lname,'Real_B']) for uname in top_names_slide]

#     # keys = lname.split('_')
    
#     # csv_dict['Action Space'].append(keys[0])
#     # csv_dict['State Space'].append(keys[1])
#     # csv_dict['Upper Folder'].append(uname)
#     # print(A_path)

#     asterisk_radar= backend.draw_radar(f1,'FTP_S1')
#     csv_dict['Sim A Distance Covered'].append(asterisk_radar[0])
#     csv_dict['Sim A Distance Std'].append(asterisk_radar[1])
#     csv_dict['Sim A Efficiency'].append(asterisk_radar[2])
#     csv_dict['Sim A Efficiency Std'].append(asterisk_radar[3])

#     asterisk_radar= backend.draw_radar(f2,'FTP_S2')
#     csv_dict['Sim B Distance Covered'].append(asterisk_radar[0])
#     csv_dict['Sim B Distance Std'].append(asterisk_radar[1])
#     csv_dict['Sim B Efficiency'].append(asterisk_radar[2])
#     csv_dict['Sim B Efficiency Std'].append(asterisk_radar[3])

#     asterisk_radar= backend.draw_radar(f3,'FTP_S3')
#     csv_dict['Real A Distance Covered'].append(asterisk_radar[0])
#     csv_dict['Real A Distance Std'].append(asterisk_radar[1])
#     csv_dict['Real A Efficiency'].append(asterisk_radar[2])
#     csv_dict['Real A Efficiency Std'].append(asterisk_radar[3])

#     backend.fig.savefig(o + '_FTP.png',dpi=400)
#     backend.reset()
#     backend.clear_axes()


# df = pd.DataFrame(csv_dict)
# df.to_csv("./sim_real_comparison_b.csv", index=False)

# csv_dict = {'Action Space':[], 'State Space':[], 'Sim A Distance Error':[], 'Sim A Distance Std':[],
#  'Sim A Orientation Error':[],'Sim A Orientation Std':[], 'Sim A Orientation Traveled':[], 'Sim A Traveled Std':[],'Real A Distance Error':[], 'Real A Distance Std':[],
#  'Real A Orientation Error':[],'Real A Orientation Std':[], 'Real A Orientation Traveled':[], 'Real A Traveled Std':[],'Sim B Distance Error':[], 'Sim B Distance Std':[],
#  'Sim B Orientation Error':[],'Sim B Orientation Std':[], 'Sim B Orientation Traveled':[], 'Sim B Traveled Std':[],'Real B Distance Error':[], 'Real B Distance Std':[],
#  'Real B Orientation Error':[],'Real B Orientation Std':[], 'Real B Orientation Traveled':[], 'Real B Traveled Std':[]}



# Begin real world comparison table making
# sub_names =['FTP_S1','FTP_S2','FTP_S3','JA_S1','JA_S2','JA_S3']
# top_names_slide = ['Mothra_Slide','HPC_Slide','Misc_Slide']
# top_names_rotate = ['Mothra_Rotation','HPC_Rotation','Jeremiah_Rotation']
# top_names_full = ['Mothra_Full','HPC_Full','Jeremiah_Full']

# backend = data_gui_backend.PlotBackend('./data/Mothra_Rotation/FTP_S1')
# for lname in sub_names:
#     A_path = ['/'.join(['./data',uname,lname,'Ast_A']) for uname in top_names_rotate]
#     B_path = ['/'.join(['./data',uname,lname,'Ast_B']) for uname in top_names_rotate]
#     Real_A_path = ['/'.join(['./data',uname,lname,'Real_A']) for uname in top_names_rotate]
#     Real_B_path = ['/'.join(['./data',uname,lname,'Real_B']) for uname in top_names_rotate]

#     keys = lname.split('_')
    
#     csv_dict['Action Space'].append(keys[0])
#     csv_dict['State Space'].append(keys[1])

#     try:
#         asterisk_radar= backend.draw_rotation_asterisk(A_path,'Sim A')
#         csv_dict['Sim A Distance Error'].append(asterisk_radar[0])
#         csv_dict['Sim A Distance Std'].append(asterisk_radar[1])
#         csv_dict['Sim A Orientation Error'].append(asterisk_radar[2])
#         csv_dict['Sim A Orientation Std'].append(asterisk_radar[3])
#         csv_dict['Sim A Orientation Traveled'].append(asterisk_radar[4])
#         csv_dict['Sim A Traveled Std'].append(asterisk_radar[5])
#     except:
#         print(f'error in Sim A, subname: {lname}, continuing crunching')
#     try:
#         asterisk_radar= backend.draw_rotation_asterisk(B_path,'Sim B')
#         csv_dict['Sim B Distance Error'].append(asterisk_radar[0])
#         csv_dict['Sim B Distance Std'].append(asterisk_radar[1])
#         csv_dict['Sim B Orientation Error'].append(asterisk_radar[2])
#         csv_dict['Sim B Orientation Std'].append(asterisk_radar[3])
#         csv_dict['Sim B Orientation Traveled'].append(asterisk_radar[4])
#         csv_dict['Sim B Traveled Std'].append(asterisk_radar[5])
#     except:
#         print(f'error in Sim B, subname: {lname}, continuing crunching')
#     try:
#         asterisk_radar= backend.draw_rotation_asterisk(Real_A_path,'Real A')
#         csv_dict['Real A Distance Error'].append(asterisk_radar[0])
#         csv_dict['Real A Distance Std'].append(asterisk_radar[1])
#         csv_dict['Real A Orientation Error'].append(asterisk_radar[2])
#         csv_dict['Real A Orientation Std'].append(asterisk_radar[3])
#         csv_dict['Real A Orientation Traveled'].append(asterisk_radar[4])
#         csv_dict['Real A Traveled Std'].append(asterisk_radar[5])
#     except:
#         print(f'error in Real A, subname: {lname}, continuing crunching')
#     try:
#         asterisk_radar= backend.draw_rotation_asterisk(Real_B_path,'Real B')
#         csv_dict['Real B Distance Error'].append(asterisk_radar[0])
#         csv_dict['Real B Distance Std'].append(asterisk_radar[1])
#         csv_dict['Real B Orientation Error'].append(asterisk_radar[2])
#         csv_dict['Real B Orientation Std'].append(asterisk_radar[3])
#         csv_dict['Real B Orientation Traveled'].append(asterisk_radar[4])
#         csv_dict['Real B Traveled Std'].append(asterisk_radar[5])
#     except:
#         print(f'error in Real B, subname: {lname}, continuing crunching')
#     backend.reset()

# df = pd.DataFrame(csv_dict)
# df.to_csv("./sim_real_comparison_rotate.csv", index=False)
