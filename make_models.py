#!/usr/bin/env python3

import subprocess
import os
import sys
import json

def run_blender(object_list='cuboid_cylinder', blender_loc = ""):
    """
    generates the hands litsted in the hand que and the given objects for each hand
    Input: object_list: as string with cuboid, cylinder or cuboid_cylinder
    """

<<<<<<< Updated upstream
    print(sys.platform)
    print(blender_loc)

    subprocess.run(f'{blender_loc} InitialFile.blend --background --python generator.py {object_list}', shell=True)

    # if sys.platform == "linux" or "linux2":
    #     # if running linux fill in the path to blender
    #     # subprocess.run(f'/snap/user/ InitialFile.blend --background --python generator.py {object_list}', shell=True)
    #     pass
    # elif sys.platform == "darwin":
    #     # macOS location for creator(Josh)
    #     subprocess.run(f'{blender_loc} InitialFile.blend --background --python generator.py {object_list}', shell=True)
    
    # elif sys.platform == "win32":
    #     print('''\n  
    #     ***********************************************************************************************

    #                                 Windows not currently supported 
=======
    # linux
    subprocess.run(f'/snap/bin/blender InitialFile.blend --background --python generator.py {object_list}', shell=True)
    # mac os
    # subprocess.run(f'/Applications/Blender.app/Contents/MacOS/Blender InitialFile.blend --background --python generator.py {object_list}', shell=True)
>>>>>>> Stashed changes

    #     ***********************************************************************************************
    #     \n''')

def read_json(user_info_loc):

    with open(user_info_loc, "r") as read_file:
        location = json.load(read_file)
    return location["blender_loc"]

if __name__ == '__main__':

    # Grabs the location and moves to the blender resources folder
    # dir = os.path.dirname(__file__)
<<<<<<< Updated upstream
=======
    dir = os.getcwd()
    os.chdir(f'{dir}/blender_resources')
>>>>>>> Stashed changes

    dir = os.getcwd()  # worked better on linux
    print(f'\n\n {dir} \n\n')
    user_info_loc = f'{dir}/User_Info.json'
    os.chdir(f'{dir}/blender_resources')
    blender_loc = read_json(user_info_loc)
    run_blender(blender_loc=blender_loc)
