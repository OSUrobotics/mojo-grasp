# @Time : 6/6/2023 11:15 AM
# @Author : Alejandro Velasquez

## --- Standard Library Imports
import os
import time
from matplotlib import pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np

def running_plot_and_video(filename_list, x_data, y_data, xlabel='Elapsed Timesteps', ylabel='Actor Output', title='Actor Output'):
    """
    Plots a graph with a vertical line running while playing a video given images
    @param filename: list of fimage filepaths
    @param x_data: data for plot's x-axis
    @param y_data: data for plot's y-axis
    @return:

    Ref:https://stackoverflow.com/questions/61808191/is-there-an-easy-way-to-animate-a-scrolling-vertical-line-in-matplotlib
    """

    # Sort png files in a list
    
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4.8))
    if len(np.shape(y_data)) >1:
        for i in range(np.shape(y_data)[1]):
            ax[0].plot(x_data,y_data[:,i],linewidth=2)
    else:
        ax[0].plot(x_data, y_data, 'k-', linewidth=2)
    ax[0].set_xlim(0, max(x_data))
    ax[0].set_ylim(0, 120)
    ax[0].grid()
    ax[0].set_xlabel(xlabel)
    ax[0].set_ylabel(ylabel)
    plt.ion()
    plt.show()
    plt.title(title, loc='right')

    # Remove details from ax[1] because we are displaying only the image
    ax[1].xaxis.set_visible(False)
    ax[1].yaxis.set_visible(False)
    for spine in ['top', 'right', 'left', 'bottom']:
        ax[1].spines[spine].set_visible(False)

    # out = None
    counter = 0
    for i,filepath in enumerate(filename_list) :
        # Vertical Line moving along the x axis
        x = i / 151
        line = ax[0].axvline(x=x, color='red', linestyle='dotted', linewidth=2)

        # Picture from the rosbag file
        img = plt.imread(filepath, 0)
        im = OffsetImage(img, zoom=0.55)
        ab = AnnotationBbox(im, (0, 0), xycoords='axes fraction', box_alignment=(0, 0))
        ax[1].add_artist(ab)
        plt.pause(0.0001)

        # # Save the figure window into an avi file
        # img = pyautogui.screenshot()
        # frame = np.array(img)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # out,write(frame)

        # if out is None:
        #     out = cv2.VideoWriter(location + filename + '/trial.avi', cv2.VideoWriter_fourcc(*'MP4V'), 40, (640, 480))
        # img_for_video = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
        # out.write(img_for_video)

        # Remove annotations to avoid RAM memory consumption
        ab.remove()
        line.remove()

        if counter == 0:
            time.sleep(10)

        counter += 1

    # out.release()


def main():
    print('nope')


if __name__ == '__main__':
    main()

# todo: same function but saving video