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
for j in range(NUM_POINTS):
    theta = random.uniform(0, 2*np.pi)
    rand_r = 1-(random.uniform(0, 0.95))**2
    rx = rand_r* 60/1000
    ry = rand_r * 40/1000

    x = rx * np.sin(theta)
    y = ry * np.cos(theta)
    x_pts.append(x)
    y_pts.append(y)
    orientation = np.random.uniform(-50/180*np.pi+0.1, 50/180*np.pi-0.1)
    orientation = orientation + np.sign(orientation)*0.1
    angs.append(orientation)
finger1 = np.random.uniform(-0.01,0.01,NUM_POINTS)
finger1 = list(finger1)
finger2 = np.random.uniform(-0.01,0.01,NUM_POINTS)
finger2 = list(finger2)
df = pd.DataFrame(
    {'x': x_pts,
      'y': y_pts,
      'ang':angs,
      'f1y':finger1,
      'f2y':finger2
      })
df.to_csv("resources/rotation_only_train.csv", index=False)            
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
plt.scatter(range(len(angs)),angs)
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