import bpy
import os


class functions():
    """
    Useful functions for creating multiple models in blender
    """

    def __init__(self, current_directory, export_directory):

        self.directory = current_directory
        self.export_directory = export_directory
        self.sim_dir = ''
        self.obj_dir = ''

    def delete_all(self):
        """
        Deletes all objects in the blender enviroment
        """
        bpy.ops.object.select_all(action='SELECT')  #deletes everything
        bpy.ops.object.delete(use_global=False)
    
    def get_part(self, name, location):
        """
        Imports base objects from secondary blender file
        Inputs: name: object name
                location: where to place the object in main blender enviroment
        """
        directory = self.directory + '/GripperComponents.blend/Object/' 
        bpy.ops.wm.append(filename=name, directory=directory)
        bpy.context.view_layer.objects.active = bpy.data.objects[name] 
        bpy.context.object.location = location
    
    def scale_part(self, name, scale):
        """
        Scales a given object
        Inputs: name: object name
                scale: x,y,z values to scale object
        """
        bpy.context.view_layer.objects.active = bpy.data.objects[name] 
        bpy.context.object.scale = scale

    def join_parts(self, names, new_name):
        """
        Combine multiple objects together
        Inputs: names: the names of the objects to be combined
                new_name: what to name the new object
        """

        for i in range(len(names) - 1):
            bpy.data.objects[names[i]].select_set(True)
        
        bpy.context.view_layer.objects.active = bpy.data.objects[names[-1]]
        bpy.ops.object.join()
        bpy.context.selected_objects[0].name = new_name

    def change_name(self, old_name, new_name):
        """
        change the name of an object
        Inputs: old_name: name of object you want to change
                new_name: name you want the object to have
        """
        bpy.context.view_layer.objects.active = bpy.data.objects[old_name]
        bpy.context.selected_objects[0].name = new_name

    def export_part(self, name):
        """
        export the object as an obj file
        Input: name: name of object you wish to export
        """
        name += '.obj'
        target_file = os.path.join(self.export_directory, name)
        bpy.ops.export_scene.obj(filepath=target_file, use_triangles=True, path_mode='COPY')

    def import_part(self, file_name, position=None, rotation=None, file_location=None):
        
        if file_location == None:
            file_location = self.export_directory
        
        part_name = file_name
        file_name += '.obj'
        # file_name += '.stl'
        target_file = os.path.join(file_location, file_name)
        bpy.ops.import_scene.obj(filepath=target_file)

        parts_list = []
        for i in range(len((bpy.context.selected_objects))):
            parts_list.append(bpy.context.selected_objects[i].name)
        self.join_parts(parts_list, part_name)

        if position != None:
            self.translate_part(part_name, position)
        if rotation != None:
            self.rotate_part(part_name, rotation)


    def rotate_part(self, part_name, rpy):

        bpy.context.view_layer.objects.active = bpy.data.objects[part_name]
        bpy.context.object.rotation_euler = rpy

    def translate_part(self, part_name, xyz):
        bpy.context.view_layer.objects.active = bpy.data.objects[part_name]
        bpy.context.object.location = xyz

    def set_directories(self, sim='', obj=''):
        """
        Change directories
        Inputs: sim: simulator directory
                obj: object directory
        """

        self.sim_dir = sim
        self.obj_dir = obj