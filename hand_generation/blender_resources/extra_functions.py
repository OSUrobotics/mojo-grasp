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

    def channge_name(self, old_name, new_name):
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

    def import_part(self, file_name, position, rotation, file_location=None):
        
        if file_location == None:
            file_location = self.export_directory
        
        file_name += '.obj'
        target_file = os.path.join(file_location, file_name)
        bpy.ops.import_scene.obj(filepath=target_file)
        bpy.context.view_layer.objects.active = bpy.data.objects[f'{file_name}_Cube']
        bpy.context.object.location = position
        bpy.context.object.rotation_euler = rotation

    def set_directories(self, sim='', obj=''):
        """
        Change directories
        Inputs: sim: simulator directory
                obj: object directory
        """

        self.sim_dir = sim
        self.obj_dir = obj