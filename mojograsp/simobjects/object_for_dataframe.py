#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 12:48:27 2022

@author: orochi
"""

from mojograsp.simobjects.object_base import ObjectBase

class ObjectVelocityDF(ObjectBase):
    def __init__(self, id: int = None, path: str = None, name: str = None):
        super().__init__(id=id, path=path, name=name)

    def get_data(self):
        """
        It is used in :func:`~mojograsp.simcore.state.StateDefault` to collect the state information 
        of an object. The default dictionary that is returned contains the current pose of the object.

        :return: dictionary of data about the object (can be used with the default state class)
        :rtype: dict
        """
        data = {}
        temp_p = self.get_curr_pose()
        data["pose-position"] = temp_p[0]
        data["pose-orientation"] = temp_p[1]
        temp_v = self.get_curr_velocity()
        data["velocity-position"] = temp_v[0]
        data["velocity-orientation"] = temp_v[1]
        return data