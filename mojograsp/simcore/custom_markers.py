from matplotlib.path import Path
import numpy as np



def rotate_by_theta(theta,xylist):
    new_x = np.cos(theta/180*np.pi)*xylist[0] - np.sin(theta/180*np.pi)*xylist[1]
    new_y = np.cos(theta/180*np.pi)*xylist[1] + np.sin(theta/180*np.pi)*xylist[0]
    return(new_x,new_y)


def custom_hemi(theta=0):
    verts = [
        (0,0), #Right middle
        (-1*np.sin(45/180*np.pi),-1*np.cos(45/180*np.pi)), #lower left
        (-1.25,0), 
        (-1*np.sin(45/180*np.pi),1*np.cos(45/180*np.pi)), #upper left 
        (0,0), #upper right
    ]

    codes = [
        Path.MOVETO, #begin the figure in the lower right
        Path.LINETO, #start a 3 point curve with the control point in lower left
        Path.CURVE3, #end curve in the upper left
        Path.LINETO, #start a new 3 point curve with the upper right as a control point
        Path.CLOSEPOLY
    ]
    test_verts = [rotate_by_theta(theta,i) for i in verts]
    cmarker = Path(test_verts,codes)
    return cmarker

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    plt.plot([0,1],[0,1],marker=custom_hemi(),markersize=200,markerfacecolor = 'b',
            markeredgecolor = 'w',linewidth=0)
    plt.plot([0,1],[0,1],marker=custom_hemi(90),markersize=200,markerfacecolor = 'r',
            markeredgecolor = 'w',linewidth=0)
    plt.plot([0,1],[0,1],marker=custom_hemi(180),markersize=200,markerfacecolor = 'r',
            markeredgecolor = 'w',linewidth=0)
    plt.plot([0,1],[0,1],marker=custom_hemi(270),markersize=200,markerfacecolor = 'r',
            markeredgecolor = 'w',linewidth=0)
    # plt.plot('.')
    plt.show()