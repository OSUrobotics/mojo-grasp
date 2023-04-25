import random
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np

# Number of points to sample in an ellipse
NUM_POINTS = 500
NUM_TRAIN = int(NUM_POINTS*0.7)
NUM_TEST = NUM_POINTS - NUM_TRAIN
# maximums for x and y
ELLIPSE_X_MAX = 83.8
ELLIPSE_Y_MAX = 55.4

x_max = ELLIPSE_X_MAX * .001
y_max = ELLIPSE_Y_MAX * .001
x_train = []
y_train = []

x_test = []
y_test = []
bin_width = 360/16
bin_start = bin_width/2
add_test = True
add_train = True
# Sample n number of points
for i in range(NUM_POINTS):
    in_ellipse = False
    while not in_ellipse:
        x = random.uniform(-x_max, x_max)
        y = random.uniform(-y_max, y_max)
        # We are sampling a rectangular area, we check to see if the point given is within the ellipse
        # If it is we add it to the list, if not we retry
        if ((x**2/((x_max*2)/2)**2) + (y**2/((y_max*2)/2)**2) < 1):
            temp = np.arctan(y/x)*180/np.pi
            if add_test and (np.floor((temp-bin_start)/bin_width) % 2 ==0):
                print(temp)
                x_test.append(x)
                y_test.append(y)
                in_ellipse = True
                if len(x_test) >= NUM_TEST:
                    add_test = False
            elif add_train and (np.floor((temp-bin_start)/bin_width) % 2 == 1):
                x_train.append(x)
                y_train.append(y)
                in_ellipse = True
                if len(x_train) >= NUM_TRAIN:
                    add_train = False

# create dataframe from lists
df = pd.DataFrame(
    {'x': x_train,
     'y': y_train
     })
# save to csv
df.to_csv("resources/train_points.csv", index=False)

# create dataframe from lists
df2 = pd.DataFrame(
    {'x': x_test,
     'y': y_test
     })
df2.to_csv("resources/test_points.csv")

# Plot the resulting samples
fig = plt.gcf()
ax = fig.gca()
ellipseL = matplotlib.patches.Ellipse(
    [0, 0], x_max*2, y_max*2, angle=0, edgecolor="black", lw=2, facecolor="none")
ax.add_patch(ellipseL)
plt.scatter(x_train, y_train)
plt.scatter(x_test, y_test)
plt.legend(['train','test'])
plt.xlim([-0.09,0.09])
plt.ylim([-0.09,0.09])
plt.show()
print(len(x_test), len(x_train))