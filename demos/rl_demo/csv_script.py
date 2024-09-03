import pandas as pd

from mojograsp.simcore.data_gui_backend import *

names = ['JA_S1','JA_S2','JA_S3','FTP_S1','FTP_S2','FTP_S3']

# upper_names=['Mothra_Slide', 'Misc_Slide', 'HPC_Slide_evaluated']

upper_names = ['Mothra_Rotation','HPC_Rotation','Jeremiah_Rotation']

# parent_folder = '/media/mothra/Samsung_T5/Analyzed Folders'

parent_folder = '/home/mothra/mojo-grasp/demos/rl_demo/data'

# eval_types = ['Eval_A', 'Eval_B']
eval_types = ['Ast_A','RB3']
temp = PlotBackend()

# temp.try_fuckery('/media/mothra/Samsung_T5/Analyzed Folders/csv_files/Misc_Slide_JA_S3_Eval_A_data.csv','/media/mothra/Samsung_T5/Analyzed Folders/csv_files/Misc_Slide_JA_S3_Eval_B_data.csv',[0.005,1000])
# input('did the fuckery')
# for upper in upper_names:
# for pol in names:
#     # a_path = parent_folder + upper+'_'+pol+'_'+eval_types[0]+'_data.csv'
#     # b_path = parent_folder + upper+'_'+pol+'_'+eval_types[1]+'_data.csv'
#     a_path = ['/'.join([parent_folder,upper_names[0],pol,eval_types[0]]),
#               '/'.join([parent_folder,upper_names[1],pol,eval_types[0]]),
#               '/'.join([parent_folder,upper_names[2],pol,eval_types[0]])]
#     b_path = ['/'.join([parent_folder,upper_names[0],pol,eval_types[1]]),
#               '/'.join([parent_folder,upper_names[1],pol,eval_types[1]]),
#               '/'.join([parent_folder,upper_names[2],pol,eval_types[1]])]
#     temp.try_fuckery(a_path,b_path,[0.005,1000],pol+'_merged.csv', '/home/mothra/mojo-grasp/demos/rl_demo/data/'+pol+'_merged.csv')
#     temp.fig.savefig(pol+'plotted.png')
#     temp.reset()
    # assert 1==0
temp.real_world_flag = True
for pol in names:
    print(pol)
    a_path = ['/'.join([parent_folder,upper_names[0],pol,eval_types[0]]),
              '/'.join([parent_folder,upper_names[1],pol,eval_types[0]]),
              '/'.join([parent_folder,upper_names[2],pol,eval_types[0]])]
    b_path = ['/'.join([parent_folder,upper_names[0],pol,eval_types[1]]),
              '/'.join([parent_folder,upper_names[1],pol,eval_types[1]]),
              '/'.join([parent_folder,upper_names[2],pol,eval_types[1]])]
    # print(a_path,b_path)
    temp.rotation_fuckery(a_path,b_path,[0.026,5],pol+'_merged_rotation_reals.csv')
    temp.reset()