import zipfile
import os
from mojograsp.simcore.data_gui_backend import *
import shutil
# Define paths
high_levels = ['Mothra_Rotation']
low_level = ['FTP_S1', 'FTP_S2', 'FTP_S3', 'JA_S1','JA_S2', 'JA_S3']
zip_file_extension = '_Test.zip'
extracted_folder = 'extracted_data'



for name in low_level:
    base_path =  '/media/mothra/Samsung_T5/Analyzed Folders/Mothra_Slide/'+name + '/'
    zip_file_path = base_path+name + zip_file_extension

    # Step 1: Extract the contents of the zip file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(base_path+extracted_folder)
    print('extracted ', name)
    # Step 2: Process each pickle file in the extracted folder
    backend = PlotBackend()
    backend.draw_relative_reward_strength(base_path + extracted_folder + '/Test/', {'DISTANCE_SCALING':0.1,'ROTATION_SCALING':1,'CONTACT_SCALING':0.2})
    backend.save_point_dictionary(base_path, name+'_combined_test_data')

    shutil.rmtree(base_path + extracted_folder)
    print('removed extracted in ', name)
