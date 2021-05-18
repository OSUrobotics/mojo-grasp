#! /usr/bin/env python3

import os
import json
import sys


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


    



if __name__ == '__main__':

    main()