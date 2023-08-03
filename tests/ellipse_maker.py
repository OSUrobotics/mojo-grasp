import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import time

with open('/home/mothra/mojo-grasp/demos/rl_demo/data/ftp_eval_evec/Train/episode_1.pkl','rb') as file:
    data = pkl.load(file)


def find_ellipse(t1,t2):
    theta = np.linspace(0,2*np.pi,1000)
    t1_range = np.sin(theta) * 0.01
    t2_range = np.cos(theta) * 0.01
    xs = []
    ys = []
    for dt1,dt2 in zip(t1_range,t2_range):
        x = np.sin(t1+dt1)*0.072 + np.sin(t1+dt1+t2+dt2)*0.072
        y = np.cos(t1+dt1)*0.072 + np.cos(t1+dt1+t2+dt2)*0.072
        xs.append(x)
        ys.append(y)
    print((np.max(xs)-np.min(xs))/(np.max(ys)-np.min(ys)))
    print()
    plt.plot(xs,ys)
    # plt.plot(t1_range,t2_range)
    # plt.xlim(-0.01,0.01)
    # plt.ylim(0.09,0.11)
    plt.show()

find_ellipse(-0.725,1.45)
# time.sleep()
for tstep in data['timestep_list']:
    eigenvalues = np.array(tstep['state']['two_finger_gripper']['eigenvalues'][2:4])
    eigenvectors = np.array(tstep['state']['two_finger_gripper']['eigenvectors'][4:8]).reshape(2,2)
    print(eigenvalues)
    print(eigenvectors)
    theta = np.linspace(0, 2*np.pi, 1000)
    ellipsis = (np.sqrt(np.abs(eigenvalues[None,:])) * eigenvectors) @ [np.sin(theta), np.cos(theta)]
    temp = np.max(ellipsis,axis=1)-np.min(ellipsis,axis=1)
    print(temp[0]/temp[1])
    plt.plot(ellipsis[0,:], ellipsis[1,:])
    # plt.xlim(-0.01,0.01)
    # plt.ylim(0.09,0.11)
    plt.show()
    


