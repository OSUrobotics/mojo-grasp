#!/usr/bin/env python3

import os
import json
import sys
import glob


def main():
    direct = os.getcwd()
    print(direct)
    
    try:
        with open(f'{direct}/User_Info.json','r') as read_file:
            location = json.load(read_file)
        
    except FileNotFoundError:
        # print("worked")

        blend_loc = input('Enter the path to blender:    ')
        temp_dect = {'blender_loc' : blend_loc}
        

        with open(f'{direct}/User_Info.json','w') as write_file:
            json.dump(temp_dect, write_file, indent=4)
    

    # direct_queue = f'{direct}/'
    # direct_archive = 
    
    if os.path.isdir(f'{direct}/hand_models/hand_queue_json/') and os.path.isdir(f'{direct}/hand_models/hand_archive_json/'):
        print('works')

    # try:
    #     os.path.isdir
    # except FileNotFoundError:
    #     pass


    



if __name__ == '__main__':

    main()