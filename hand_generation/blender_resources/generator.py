#!/usr/bin/env python3

import os
import sys
import bpy
import csv
import json
import glob
from pathlib import Path
from numpy import sin, cos, pi
import math



class Generator():
    """
    Creates the hands that are in the que
    """

    def __init__(self, hand_info, object_list, json_filename):

        func = bpy.data.texts["extra_functions.py"].as_module()
        

        urdf_create = bpy.data.texts["urdf.py"].as_module()

        self.json_filename = json_filename
        print(f' \n\n\n {json_filename} \n\n\n')
        self.current_directory = os.path.dirname(__file__)  
        self.main_directory, _ = os.path.split(self.current_directory)
        self.hand_directory = f'{self.main_directory}/hand_models'
        self.archive_directory = f'{self.main_directory}/hand_models/hand_archive_json'
        self.object_directory = f'{self.main_directory}/object_models'

        self.left_finger = hand_info["left_finger"]
        self.right_finger = hand_info["right_finger"]
        self.palm = hand_info["palm"]
        self.hand_name = hand_info["hand_name"]

        self.object_list = object_list

        self.scale_factor = hand_info["hand_parameters"]["scale_factor"]
        self.palm_width = hand_info["hand_parameters"]["palm_width"] * self.scale_factor
        self.palm_height = hand_info["hand_parameters"]["palm_height"] * self.scale_factor
        self.hand_thickness = hand_info["hand_parameters"]["hand_thickness"] * self.scale_factor
        self.finger_width = hand_info["hand_parameters"]["finger_width"] * self.scale_factor
        self.joint_length = hand_info["hand_parameters"]["joint_length"] * self.scale_factor

        self.object_heigt = self.hand_thickness * 3

        hand_mesh_dir = f'{self.hand_directory}/{self.hand_name}/obj_files/'
        self.object_mesh_dir = f'{self.object_directory}/{self.hand_name}/obj_files/'
        self.create_folders(hand_mesh_dir)
        self.create_folders(self.object_mesh_dir)

        self.functions = func.functions(current_directory=self.current_directory, export_directory=hand_mesh_dir)
        self.functions.delete_all()

        self.urdf = urdf_create.URDF_Generator(f'{self.hand_directory}/{self.hand_name}/')

        self.parent_height = 0

        self.grasp_width = 0

        self.palm_name = ''

        self.create_hand()
        self.move_json(hand_info, json_filename)

    def create_folders(self, dir):
        """
        Creates new folders for the hand models and object models
        Input: dir: Folder directory for the folder to be created
        """

        Path(dir).mkdir(parents=True, exist_ok=True)

    def create_hand(self):
        """
        Main code block that calls the functions to generate individual components
        """
        # self.functions.set_directories(obj=f'{self.hand_directory}/obj_files/')
        self.urdf.start_file(self.hand_name)
        self.create_palm()
        self.create_finger(self.left_finger, 'l')
        self.create_finger(self.right_finger, 'r')
        self.urdf.end_file()
        self.urdf.write(filename=self.hand_name)
        self.create_object()

    def create_palm(self):
        """
        Creates the palm for the hand
        """

        # set up palm name and collision varient
        self.palm_name = f'body_palm_l{self.palm["left_joint"]}vr{self.palm["right_joint"]}'
        collision_name = f'{self.palm_name}_collision'

        # import the palm model set it at the origin, scale it, rename it for collision and export obj file
        self.functions.get_part("palm", (0, 0, 0))
        self.functions.scale_part("palm", (self.hand_thickness, (self.palm_width + self.finger_width), self.palm_height))
        self.functions.change_name("palm", collision_name)
        self.functions.export_part(collision_name)
        # change object to base palm name
        self.functions.change_name(collision_name, self.palm_name)

        y_loc = (self.palm_width / 2)  # y component of finger location
        # Grab the left and right joints, set location, scale and rename
        self.functions.get_part(self.palm["left_joint"], (0, -y_loc, self.palm_height))
        self.functions.scale_part(self.palm["left_joint"], (self.hand_thickness, self.finger_width, (self.finger_width / 2)))
        self.functions.change_name(self.palm["left_joint"], "left_joint")
        self.functions.get_part(self.palm["right_joint"], (0, y_loc, self.palm_height))
        self.functions.change_name(self.palm["right_joint"], "right_joint")
        self.functions.scale_part("right_joint", (self.hand_thickness, self.finger_width, (self.finger_width / 2)))

        # combine the palm and joints, export obj file
        self.functions.join_parts(('right_joint', 'left_joint', self.palm_name), self.palm_name)
        self.functions.export_part(self.palm_name)

        # add the palm link to the urdf file
        self.urdf.link(name=self.palm_name, pose=(0, 0, 0, 0, (pi / 2), 0), scale=(1, 1, 1), mass=0.5)

        # remove objects for the next component
        self.functions.delete_all()


    def create_finger(self, finger, side):
        """
        Generates the segments for the left or right finger
        Inputs: finger: dictionary that describes the finger
                side: l or r (left or right finger)
        """

        # gets base info about finger
        number_segments = finger["number_segments"]
        # finger length is the total finger length minus the length from the joints
        finger_length = finger["finger_length"] - (self.joint_length * number_segments)

        # start at bottom and work your way up
        self.parent_height = 0
        parent_name = self.palm_name

        # creates the name and calls function to generate each segment in the finger
        for seg_name in finger["segment_names"]:
            if seg_name == "distal":  # distals have some different info in them for naming
                child_name = f'body_{side}_{seg_name}_{finger[seg_name]["joint_bottom"]}v{finger[seg_name]["ending"]}'

            else:
                child_name = f'body_{side}_{seg_name}_{finger[seg_name]["joint_bottom"]}v{finger[seg_name]["joint_top"]}'

            # calls function for creating the segment
            self.create_finger_segment(parent_name, segment=finger[seg_name], finger_length=finger_length, side=side, child_name=child_name, seg_name=seg_name)
            
            parent_name = child_name  # set the current segment name(child) to parent


    def create_sensors(self, segment, seg_len, side, parent):
        """
        If the segment is the distal segment add sensors to it
        Input: segment: dictionary that describes the segment
               seg_len: how long the segment is
               side: l or r (left or right side)
               parent: name of parent, which is the distal segment for the sensors
        """

        # gets the sensor locate, scale and export it
        self.functions.get_part("sensor", (0, 0, 0))
        self.functions.scale_part("sensor", (1, .1, .1))
        self.functions.export_part("sensor")
        self.functions.change_name("sensor", "sensor_collision")
        self.functions.export_part("sensor_collision")

        # depending on side the offset switches along the x-axis
        if side == "l":
            x_cooridnate = self.finger_width / 2
        else:
            x_cooridnate = -(self.finger_width / 2)

        sensor_number = segment["sensors"]["num"]
        half = math.ceil(sensor_number / 2)  # sensors are placed in two columns
        height = seg_len / (half + 1)  # the height spacing for the sensors
        z_coordinate = self.hand_thickness / 4  # get the z-offset
        
        y_coordinate = self.joint_length + (height * half)  # starting for y-coordinate

        # for the number of sensors locate and add link/joint to urdf file
        for i in range(sensor_number):  # gets the location of the sensor
            if i % 2 == 0:
                if i == (sensor_number - 1):  # places an odd number of sensors with the last one in the center
                    xyz = (x_cooridnate, y_coordinate, 0)
                else:
                    xyz = (x_cooridnate, y_coordinate, z_coordinate)
            else:
                xyz = (x_cooridnate, y_coordinate, -z_coordinate)
                y_coordinate -= height

            joint_name = f'{side}_sensor_{i}'  # creates sensor name and adds link/joint to urdf
            self.urdf.joint(joint_name, Type="fixed", child=f'{side}_sensor_{i}', parent=parent, rpy_in=(0,0,0), xyz_in=xyz)
            if side == "l":
                self.urdf.link(name=f'{side}_sensor_{i}', pose=(0, 0, 0, 0, 0, 0), scale=(1, 1, 1), mass=0.5, model_name="sensor")
            else:
                self.urdf.link(name=f'{side}_sensor_{i}', pose=(0, 0, 0, 0, 0, pi), scale=(1, 1, 1), mass=0.5, model_name="sensor")
        
        self.functions.delete_all()  # clear the objects


    def create_finger_segment(self, parent, segment, finger_length, side, child_name, seg_name):
        """
        Generates the individual segments of a finger
        Input: parent: name of parent link
               segment: dictionary describing the segment
               finger_length: how long the finger segments are
               side: l or r (left or right side)
               child_name: name of current segment for urdf/obj naming
               seg_name: distal, intermediate, proximal (what link is the segment)
        """

        if parent == self.palm_name:  # first segment on each side starts on top of the palm

            if side == "l":  # flip positive and negative depending on side
                xyz = [-(self.palm_width / 2), self.palm_height, 0]
            else:
                xyz = [(self.palm_width / 2), self.palm_height, 0]
        else:

            xyz = [0, self.parent_height, 0]  # set the segment height

        seg_len = finger_length * segment["ratio"]  # overall length times ratio for this segment
        collision_name = f'{child_name}_collision'
        rpy = [0, 0, 0]  # no rotations
        loc_top = self.joint_length + seg_len  # location for topping pieces of segment
        self.parent_height = loc_top

        if seg_name == "distal":  # distal have their toppers as ending
            topping = segment["ending"]
            # adds midpoint distal to max width value
            self.grasp_width += sin(30 * pi / 180) * ((seg_len / 2) + self.joint_length)
        else:
            topping = segment["joint_top"]  # non distal it is topped with a joint
            self.grasp_width += seg_len + self.joint_length  # adding joint and segment to width

        # import, scale, join and export the topping, body and bottom portions of the segment
        self.functions.get_part(topping, (0, 0, loc_top))
        self.functions.scale_part(topping, (self.hand_thickness, self.finger_width, (self.finger_width/2)))

        self.functions.get_part('Test_Box', (0, 0, self.joint_length))
        self.functions.scale_part('Test_Box', (self.hand_thickness, self.finger_width, seg_len))
        
        self.functions.join_parts((topping, 'Test_Box'), collision_name)
        bpy.ops.object.origin_set(type='ORIGIN_CURSOR', center='MEDIAN')

        self.functions.export_part(collision_name)

        self.functions.get_part(f'{segment["joint_bottom"]}_joint', (0, 0, 0))
        self.functions.scale_part(f'{segment["joint_bottom"]}_joint', (self.hand_thickness, self.finger_width/2, self.joint_length))
        self.functions.join_parts((collision_name, f'{segment["joint_bottom"]}_joint'), child_name)

        self.functions.export_part(child_name)

        # add the joint and link to the urdf file
        joint_name = f'{side}_{seg_name}_{segment["joint_bottom"]}'
        self.urdf.joint(joint_name, Type="revolute", axis=(0, 0, 1), child=child_name, parent=parent, rpy_in=rpy, xyz_in=xyz)
        self.urdf.link(name=child_name, pose=(0, 0, 0, 0, (pi / 2), 0), scale=(1, 1, 1), mass=0.5)

        self.functions.delete_all()  # clear all objects

        if seg_name == "distal":  # distal links have sensors
            if segment["sensors"]["num"] > 0:
                self.create_sensors(segment=segment, seg_len=seg_len, side=side, parent=child_name)
        

    def create_object(self):
        """
        Create the objects specific for the hand, 25%, 50%, and 75% of the max width
        """

        # get the new directory
        urdf_dir, _ = os.path.split(self.object_mesh_dir[:-1])
        urdf_dir += '/'
        # add the palm for the max width
        self.grasp_width += self.palm_width
        # set object mesh location
        self.functions.export_directory = self.object_mesh_dir
        obj_list = object_list.split('_')
        # (small, medium, large)
        # create the three sized object widths
        object_sizes = (self.grasp_width * 0.25, self.grasp_width * 0.5, self.grasp_width * 0.75)
        size_names = ('small', 'medium', 'large')

        self.urdf.dir = urdf_dir

        for obj in obj_list:  # create each object in the list
            for i in range(3):
                
                self.urdf.new_urdf()  # start a new urdf file

                # set name, import, and scale the object
                object_name = f'{self.hand_name}_{obj}_{size_names[i]}'
                self.functions.get_part(obj, (0, 0, 0))
                self.functions.scale_part(obj, (object_sizes[i], object_sizes[i], self.object_heigt))
                self.functions.change_name(obj, object_name)
                self.functions.export_part(object_name)

                object_name_collision = f'{object_name}_collision'
                self.functions.change_name(object_name, object_name_collision)
                self.functions.export_part(object_name_collision)

                self.urdf.start_file(object_name)
                self.urdf.link(name=object_name, pose=(0, 0, 0, pi/2, 0, 0))
                self.urdf.end_file()
                self.urdf.write(object_name)  # write the urdf and clear objecs
                self.functions.delete_all()

    def move_json(self, json_data, file_loc):
        """
        Once the hand is create place a copy of the json in the hand folder, and in the archives, then remove from the
        que folder
        Input: json_data: the json info read in from the original file
               file_loc: location of original file to be deleted
        """
        _, file_name = os.path.split(file_loc)  # get the file name to use in copies
        # make copies
        with open(f'{self.hand_directory}/{self.hand_name}/{file_name}', 'w') as file:
            json.dump(json_data, file, indent=4)
        
        with open(f'{self.archive_directory}/{file_name}', 'w') as file:
            json.dump(json_data, file, indent=4)
        
        os.remove(file_loc)  # delete the original json file
        
    def hand_poses(self, finger_angles=[], part_names=[]):
        finger_angles= [90, 90, 90, 90, 90, 90]
        number_parts = len(part_names)

        for part_name, i in enumerate(number_parts):
            #testinh
            self.functions.import_part(part_name, xyz, rpy)


def read_jsons(object_list):
    """
    Reads in all the json files in the que folder
    Inputs: object_list: list of objects as a str seperated with '_'
    """

    current_directory = os.path.dirname(__file__)
    main_directory, _ = os.path.split(current_directory)
    hand_directory = f'{main_directory}/hand_models'
    hand_queue_dir = f'{hand_directory}/hand_queue_json/'

    for files in glob.glob(f'{hand_queue_dir}*.json'):
        with open(files, "r") as read_file:
            hand_info = json.load(read_file)

        Generator(hand_info, object_list, files)


if __name__ == '__main__':
    try:  # list of objects to be created are passed in as an argument
        object_list = sys.argv[-1]
    except:
        object_list = ['cuboid']
    
    print(f'\n\n\n {object_list} \n\n\n')
    
    read_jsons(object_list)