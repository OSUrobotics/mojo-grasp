# mojo-grasp (Hand Generator)

## What Does This Code Do?

The purpose for this code is automate the process of creating simplified 3D models of grippers to be used in the mojo-grasp simulator. You give the code a json file describing the gripper and the code returns 3D models of that gripper the urdf file that allows the simulator to use it as well as creating default objects scaled to the grippers size. 

## Instructions

* First clone the repo/branch that includes this code.
* Insure you have blender 2.8 or newer installed on your computer.  This tool uses blender's API to create the 3D models.
* This tool also uses python3, so when runing scripts make sure you are using python3 and not python2
* In the terminal navigate to the /hand_generation/ directory and run: 

        python3 first_run.py
    * This script will create two new directories:
        * hand_models: which contains a couple default directories used in hand generation, and all the folder for each gripper that contains the models used by the simulator.
            * hand_archive_json: Keeps a copy of each json file that is ran through the generator, so the gripper can be reproduced if needed.
            * hand_queue_json: place the json file for the gripper desing here to be created once make_models is ran.
        * object_models: Contains the folders that hold the models of the object scaled to its respective grippers size.
    * The scrpit will also ask for the location of your blender install to be used for the blender calls.
        * One way to get this is to run in the terminal of a linux machine:

                which blender
* Now you can just run the following and the default grippers will be created, or you can skip this step and go on to creating unique json files.
        
        python3 make_models.py
* To create a custom gripper there is a json_creater.py tool which helps you properly fill out the json files to create a gripper. Run the following in the terminal and following the instructions.

        python3 json_creater.py
    * This program will create a json file and place it in the hand_queue_json/ directory ready to be used once make_models.py is ran.
* If you want to look at you model in the simulator you can use the simulator.py
    * Run the following:

            python3 simulator.py
    * The script will ask for the gripper's name, object type(currently just cuboid), and object size (small, medium, or large)

