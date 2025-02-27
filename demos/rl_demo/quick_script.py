import os
import shutil
# import pickle as pkl

base_path = './data/Static_3/'
new_path = './data/bunch_of_expert_trials/'

sub_folders = ['square_A','square25_A','circle_A','circle25_A','triangle_A','triangle25_A','square_circle_A','pentagon_A']

for folder in sub_folders:
    filenames = os.listdir(base_path+folder)
    for fn in filenames:
        shutil.copyfile(base_path+folder+'/'+fn, new_path+folder+fn)