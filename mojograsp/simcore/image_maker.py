import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation as R

class ImageGenerator():
    def __init__(self, image_size, goal_size=20, object_size=40):
        self.im_size = image_size
        self.goal_size=goal_size
        self.obj_size = object_size

    def draw_stamp(self, object, goals, edges=np.array([[-0.08,0.08],[0.02,0.18]])):
        image = np.ones(self.im_size, np.uint8)*128
        # given object pose (x,y,z) (x,y,z,w) need to get a square corners and angle
        diffs = np.array([edges[0][1] - edges[0][0], edges[1][1] - edges[1][0]])
        object_pos_pixel = np.array([object[0][0]-edges[0][0],edges[1][1]- object[0][1]])/diffs*self.im_size[0:2]
        angle = R.from_quat(object[1])
        angle = angle.as_euler('xyz')[-1]
        object_corner = [[],[]]
        border_size = np.array(np.cos(angle)*self.obj_size/2,np.sin(angle)*self.obj_size/2)
        object_corner[0] = object_pos_pixel - border_size
        object_corner[1] = object_pos_pixel + border_size

        rrect = cv.RotatedRect(object_pos_pixel, [self.obj_size,self.obj_size],angle*180/np.pi)
        box = cv.boxPoints(rrect)
        box = np.int0(box)
        # Draw the filled rotated rectangle
        # print(goals)
        cv.fillPoly(image, [box], 0)
        # goals = np.reshape(goals,(5,2))
        for goal in goals:
            # print(goal)
            gp = (np.array([goal[0],edges[1][1]+edges[1][0] - goal[1]-0.1])-edges[:,0])/diffs*self.im_size[0:2]
            # print(gp)
            gp = [int(gp[0]),int(gp[1])]
            # print(type(gp[0]))
            cv.circle(image,gp,self.goal_size, 255, -1)
        return image

    def draw_fill(self,object,goal_region):
        pass

    def draw_obstacle(self,object,goal,obstacle):
        pass


if __name__  == '__main__':
    test  = ImageGenerator(np.array((240,240)))
    fig = test.draw_stamp([[0,0.1,0.6],[ 0, 0, 0, 1]],
                          [[0.03,0.024],[-0.05,0.0367],[-0.06,-0.023],[0.02,-0.05],[0.05,0.02],[-0.003,0.05]])
    cv.imshow('image',fig)
    # waits for user to press any key
    # (this is necessary to avoid Python kernel form crashing)
    cv.waitKey(0)

    # closing all open windows
    cv.destroyAllWindows()
