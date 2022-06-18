import random
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

# Number of points to sample in an ellipse
NUM_POINTS = 100
# maximums for x and y
ELLIPSE_X_MAX = 83.8
ELLIPSE_Y_MAX = 55.4

x_max = ELLIPSE_X_MAX * .001
y_max = ELLIPSE_Y_MAX * .001
x_positions = []
y_positions = []
# Sample n number of points
for i in range(NUM_POINTS):
    in_ellipse = False
    while not in_ellipse:
        x = random.uniform(-x_max, x_max)
        y = random.uniform(-y_max, y_max)
        # We are sampling a rectangular area, we check to see if the point given is within the ellipse
        # If it is we add it to the list, if not we retry
        if ((x**2/((x_max*2)/2)**2) + (y**2/((y_max*2)/2)**2) < 1):
            x_positions.append(x)
            y_positions.append(y)
            in_ellipse = True

# create dataframe from lists
df = pd.DataFrame(
    {'x': x_positions,
     'y': y_positions
     })
# save to csv
df.to_csv("resources/points.csv", index=False)

# Plot the resulting samples
fig = plt.gcf()
ax = fig.gca()
ellipseL = matplotlib.patches.Ellipse(
    [0, 0], x_max*2, y_max*2, angle=0, edgecolor="black", lw=2, facecolor="none")
ax.add_patch(ellipseL)
plt.scatter(x_positions, y_positions)
plt.show()
