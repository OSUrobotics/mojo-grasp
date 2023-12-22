import os
import shutil
from pathlib import Path

start_folder = '/media/mothra/Samsung_T5/InHandRL/finger_change_no_noise/'

end_folder = '/home/mothra/mojo-grasp/demos/rl_demo/data'

start_names = ['FTP_full',  'FTP_half',  'JA_Full',  'JA_half']

end_names = ['FTP_fullstate_A_rand','FTP_halfstate_A_rand','JA_fullstate_A_rand','JA_halfstate_A_rand']

sub_folders = ['2v2_70.30_70.30_1.1_63']
for sname,ename in zip(start_names,end_names):
    tpath = Path(start_folder).joinpath(sname,sub_folders[0])
    run_folders = os.listdir(tpath)
    for i, rf in enumerate(run_folders):
        full = tpath.joinpath(rf)
        all_files = os.listdir(full)
        all_files = [file for file in all_files if 'RL' in file]
        all_files2 = [all_files[j][:-4] + str(int(j+i*len(all_files))) + all_files[j][-4:] for j in range(len(all_files))]
        for oldfile,newfile in zip(all_files,all_files2):
            shutil.move(full.joinpath(oldfile), Path(end_folder).joinpath(ename,'Real_B',newfile))