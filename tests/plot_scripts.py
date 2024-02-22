# @Time : 6/6/2023 11:15 AM
# @Author : Alejandro Velasquez

## --- Standard Library Imports
import os
import time
from matplotlib import pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import re
import pickle as pkl

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


       
# fig, (ax1,ax2) = plt.subplots(2,1,height_ratios=[2,1])
transfer_paths = ['/home/mothra/mojo-grasp/demos/rl_demo/data/JA_fullstate_A_rand/eval_b_moving/',
                '/home/mothra/mojo-grasp/demos/rl_demo/data/FTP_fullstate_A_rand/eval_b_moving/',
                '/home/mothra/mojo-grasp/demos/rl_demo/data/JA_halfstate_A_rand/eval_b_moving/',
                '/home/mothra/mojo-grasp/demos/rl_demo/data/FTP_halfstate_A_rand/eval_b_moving/']

folder_paths = ['/home/mothra/mojo-grasp/demos/rl_demo/data/JA_fullstate_A_rand/Eval/',
                '/home/mothra/mojo-grasp/demos/rl_demo/data/FTP_fullstate_A_rand/Eval/',
                '/home/mothra/mojo-grasp/demos/rl_demo/data/JA_halfstate_A_rand/Eval/',
                '/home/mothra/mojo-grasp/demos/rl_demo/data/FTP_halfstate_A_rand/Eval/']
fig, axes = plt.subplots(4,1,sharex=True)

for folder_path, second_path, ax in zip(folder_paths, transfer_paths, axes):
    # print('need to load in episode all first')
    episode_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith('.pkl')]
    filenames_only = [f for f in os.listdir(folder_path) if f.lower().endswith('.pkl')]

    filenums = [re.findall('\d+',f) for f in filenames_only]
    final_filenums = []
    for i in filenums:
        if len(i) > 0 :
            final_filenums.append(int(i[0]))

    sorted_inds = np.argsort(final_filenums)
    final_filenums = np.array(final_filenums)
    episode_files = np.array(episode_files)
    filenames_only = np.array(filenames_only)
    episode_files = episode_files[sorted_inds].tolist()
    goals, end_dists, end_poses = [],[], []
    for episode_file in episode_files:
        with open(episode_file, 'rb') as ef:
            tempdata = pkl.load(ef)

        data = tempdata['timestep_list']
        # end_position = data[-1]['state']['obj_2']['pose'][0]
        goals.append(data[0]['state']['goal_pose']['goal_pose'][0:2])
        end_dists.append(data[-1]['reward']['distance_to_goal'])
        end_poses.append(data[-1]['state']['obj_2']['pose'][0])
    
    episode_files2 = [os.path.join(second_path, f) for f in os.listdir(second_path) if f.lower().endswith('.pkl')]
    filenames_only2 = [f for f in os.listdir(second_path) if f.lower().endswith('.pkl')]

    filenums2 = [re.findall('\d+',f) for f in filenames_only2]
    final_filenums2 = []
    for i in filenums2:
        if len(i) > 0 :
            final_filenums2.append(int(i[0]))

    sorted_inds2 = np.argsort(final_filenums2)
    final_filenums2 = np.array(final_filenums2)
    episode_files2 = np.array(episode_files2)
    filenames_only2 = np.array(filenames_only2)
    episode_files2 = episode_files2[sorted_inds2].tolist()
    goals2, end_dists2, end_poses2 = [],[], []
    for episode_file in episode_files2:
        with open(episode_file, 'rb') as ef:
            tempdata = pkl.load(ef)

        data = tempdata['timestep_list']
        # end_position = data[-1]['state']['obj_2']['pose'][0]
        goals2.append(data[0]['state']['goal_pose']['goal_pose'][0:2])
        end_dists2.append(data[-1]['reward']['distance_to_goal'])
        end_poses2.append(data[-1]['state']['obj_2']['pose'][0])

    range_width = 0.03
    num_bins = 50
    bins = np.linspace(0,range_width,num_bins) + range_width/num_bins
    num_things = np.zeros(num_bins)
    num_things2 = np.zeros(num_bins)
    small_thold = max(0.005,min(end_dists))
    med_thold = small_thold+0.005
    big_thold = med_thold + 0.01
    goals = np.array(goals)
    end_poses = np.array(end_poses) - np.array([0,0.1,0])
    for pose,dist,d2 in zip(end_poses,end_dists, end_dists2):
        a= np.where(dist<bins)
        b = np.where(d2<bins)
        try:
            num_things[a[0][0]] +=1
        except IndexError:
            # print('super far away point')
            num_things[-1] +=1
        try:
            num_things2[b[0][0]] +=1
        except IndexError:
            num_things2[-1] +=1
    ax.bar(bins, num_things, width=range_width/num_bins)
    ax.bar(bins, num_things2, width=range_width/num_bins)
    # ax.legend(['Hand A Results','Hand B Results'])
    ax.set_ylim([0,300])
plt.show()

def main():
    print('nope')


if __name__ == '__main__':
    main()

# todo: same function but saving video