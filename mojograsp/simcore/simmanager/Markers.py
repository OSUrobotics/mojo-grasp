#!/usr/bin/env python3

import pybullet as p
import time
import numpy as np
# from ObjectsInScene import SceneObject


class Marker:
    """
    A Visualization tool to help find particular points in the simulation
    """

    def __init__(self, color=None, height=0.08, length=0.005, lifetime=0, size=3):
        """
        Initialising a marker with a shape
        :param shape: Shape of visual object being created
        :param color: Color of the marker
        """
        if color is None:
            self.color = [1, 0, 0]
        else:
            self.color = color
        self.height = height
        self.length = length
        self.lifetime = lifetime
        self.size = size

    def set_marker_pose(self, pose):
        """
        Place marker at a given position and orientation
        :param pose: List containing pos and orn [(x, y, z), (r_x, r_y, r_z, w)] we want to place marker at.
        :return
        """
        ID = p.addUserDebugText(text=".", textPosition=[pose[0], pose[1], self.height], textColorRGB=self.color,
                           textSize=self.size, lifeTime=self.lifetime)
        return ID

    def reset_marker_pose(self, pose, ID, height=None, lifetime=None, size=None):
        """
        Replace a marker at  a different pose
        :param ID: ID of marker to move (gotten from set marker pose)
        :param pose: Pose to move marker to
        :return: ID: new ID
        """
        if height is None:
            height = self.height
        if lifetime is None:
            lifetime = self.lifetime
        if size is None:
            size = self.size
        ID = p.addUserDebugText(text=".", textPosition=[pose[0], pose[1], height], textColorRGB=self.color,
                                textSize=size, lifeTime=lifetime, replaceItemUniqueId=ID)
        return ID


if __name__ == "__main__":
    pass
