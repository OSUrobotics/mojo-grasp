#!/usr/bin/env python3

import subprocess
import os


def run_blender(object_list='cuboid_cylinder'):
    """
    generates the hands litsted in the hand que and the given objects for each hand
    Input: object_list: as string with cuboid, cylinder or cuboid_cylinder
    """

    subprocess.run(f'/Applications/Blender.app/Contents/MacOS/Blender InitialFile.blend --background --python generator.py {object_list}', shell=True)


if __name__ == '__main__':

    # Grabs the location and moves to the blender resources folder
    dir = os.path.dirname(__file__)
    os.chdir(f'{dir}/blender_resources')

    run_blender()
