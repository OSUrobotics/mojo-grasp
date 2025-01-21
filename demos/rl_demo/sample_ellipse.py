import random
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np

# Number of points to sample in an ellipse
NUM_POINTS = 1200
NUM_TRAIN = int(NUM_POINTS*0.7)
NUM_TEST = NUM_POINTS - NUM_TRAIN
# maximums for x and y
ELLIPSE_X_MAX = 80
ELLIPSE_Y_MAX = 50

x_max = ELLIPSE_X_MAX * .001
y_max = ELLIPSE_Y_MAX * .001

x_n = []
y_n = []

x_ne = []
y_ne = []

x_nw = []
y_nw = []

x_w = []
y_w = []

x_s = []
y_s = []

x_e = []
y_e = []

x_sw = []
y_sw = []

x_se = []
y_se = []

bin_width = 360/16
bin_start = bin_width/2
add_test = True
add_train = True
separation_lines = np.array(range(9)) * np.pi/4 - np.pi/8

dirs = [[x_e,y_e],[x_ne, y_ne],[x_n,y_n],[x_nw,y_nw],[x_w,y_w],[x_sw,y_sw],[x_s,y_s],[x_se,y_se]]

# # Sample n number of points
# for i,name in enumerate(dirs):
#     for j in range(NUM_POINTS):
#         theta = random.uniform(separation_lines[i], separation_lines[i+1])
#         r = (1-(random.uniform(0, 0.95))**2) * 35/1000
#         # We are sampling a rectangular area, we check to see if the point given is within the ellipse
#         # If it is we add it to the list, if not we retry
#         x = r * np.sin(theta)
#         y = r * np.cos(theta)
#         name[0].append(x)
#         name[1].append(y)
x_pts,y_pts, angs = [], [], []
points = [[0,0]]
n = 6
rx = 60/1000
ry = 40/1000
num_layers = 3

# for i in range(1,num_layers+1):
#     print('starting i')
#     for j in range(n*i):
#         theta = np.pi*2*(j/(n*i))
#         r = np.sqrt((i/num_layers*rx*np.cos(theta))**2 + (i/num_layers*ry*np.sin(theta))**2)
#         print('theta and r', theta, r)
#         points.append([r*np.cos(theta),r*np.sin(theta)])
    
# points = np.array(points)
# print(len(points))
# x = points[:,0]
# y = points[:,1]
# plt.scatter(points[:,0],points[:,1])
# plt.axis('square')
# plt.xlabel('X pos (m)')
# plt.ylabel('Y pos (m)')
# plt.show()
# plt.scatter(range(len(angs)),angs)

# df = pd.DataFrame(
#     {'x': x,
#       'y': y
#       })
# df.to_csv("resources/start_poses.csv", index=False)   

x2_pts =[]
y2_pts = []
for j in range(NUM_POINTS):
    theta1 = random.uniform(0, 2*np.pi)
    rand_r1 = 1 - (random.uniform(0.7, 0.95))**2
    theta2 = theta1 + random.uniform(-3*np.pi/4, 3*np.pi/4)
    rand_r2 = 1 - (random.uniform(0.1, 0.6))**2
    r1x = rand_r1 * 70/1000
    r1y = rand_r1 * 70/1000
    r2x = rand_r2 * 70/1000
    r2y = rand_r2 * 70/1000
    x1 = r1x * np.sin(theta1)
    y1 = r1y * np.cos(theta1)
    x2 = r2x * np.sin(theta2)
    y2 = r2y * np.cos(theta2)
    x_pts.append(x1)
    y_pts.append(y1)
    x2_pts.append(x2)
    y2_pts.append(y2)
    orientation = 0

    angs.append(orientation)
finger1 = np.random.uniform(-0.01,0.01,NUM_POINTS)
finger1 = list(finger1)
finger2 = np.random.uniform(-0.01,0.01,NUM_POINTS)
finger2 = list(finger2)
# x_pts = np.zeros(128)
# x_pts = list(x_pts)
# y_pts = np.zeros(128)
# y_pts = list(y_pts)
# angs = np.linspace(-50/180*np.pi,50/180*np.pi,128)
# angs = list(angs)
# finger1 = np.zeros(128)
# finger1 = list(finger1)
# finger2 = np.zeros(128)
# finger2 = list(finger2)
# print(angs)
plt.scatter(x_pts,y_pts)
plt.scatter(x2_pts,y2_pts)
plt.legend(['near','far'])
plt.show()
df = pd.DataFrame(
    {'x1': x_pts,
      'y1': y_pts,
      'x2':x2_pts,
      'y2':y2_pts,
      'ang':angs,
      'f1y':finger1,
      'f2y':finger2
      })
df.to_csv("resources/simple_HRL.csv", index=False)            
# create dataframe from lists
# df = pd.DataFrame(
#     {'x': x_n,
#      'y': y_n
#      })
# df.to_csv("resources/N_points.csv", index=False)

# df = pd.DataFrame(
#     {'x': x_ne,
#      'y': y_ne
#      })
# df.to_csv("resources/NE_points.csv", index=False)

# df = pd.DataFrame(
#     {'x': x_e,
#      'y': y_e
#      })
# df.to_csv("resources/E_points.csv", index=False)

# df = pd.DataFrame(
#     {'x': x_se,
#      'y': y_se
#      })
# df.to_csv("resources/SE_points.csv", index=False)

# df = pd.DataFrame(
#     {'x': x_s,
#      'y': y_s
#      })
# df.to_csv("resources/S_points.csv", index=False)

# df = pd.DataFrame(
#     {'x': x_sw,
#      'y': y_sw
#      })
# df.to_csv("resources/SW_points.csv", index=False)

# df = pd.DataFrame(
#     {'x': x_w,
#      'y': y_w
#      })
# df.to_csv("resources/W_points.csv", index=False)

# df = pd.DataFrame(
#     {'x': x_nw,
#      'y': y_nw
#      })
# df.to_csv("resources/NW_points.csv", index=False)

# for things in dirs:
# thetas = np.linspace(0,8*np.pi,64)
# rs = np.linspace(0,0.05,64)
# x_pts = rs*np.sin(thetas)
# y_pts = rs*np.cos(thetas)
plt.scatter(x_pts,y_pts)
plt.show()
# plt.scatter(range(len(angs)),angs)
# create dataframe from lists
# df2 = pd.DataFrame(
#     {'x': x_test,
#      'y': y_test
#      })
# df2.to_csv("resources/test_points.csv")

# # Plot the resulting samples
# fig = plt.gcf()
# ax = fig.gca()
# ellipseL = matplotlib.patches.Ellipse(
#     [0, 0], x_max*2, y_max*2, angle=0, edgecolor="black", lw=2, facecolor="none")
# ax.add_patch(ellipseL)
# plt.scatter(x_train, y_train)
# plt.scatter(x_test, y_test)
# plt.legend(['train','test'])
# plt.xlim([-0.09,0.09])
# plt.ylim([-0.09,0.09])
# plt.show()
# print(len(x_test), len(x_train))