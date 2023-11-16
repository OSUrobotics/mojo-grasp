import matplotlib.pyplot as plt
import numpy as np
import csv


with open('./resources/test_points_big.csv') as csvfile:
    reader = csv.reader(csvfile)
    a = True
    points = []
    for row in reader:
        if a:
            a=False
        else:
            points.append([float(row[0]),float(row[1])])

with open('./resources/test_points.csv') as csvfile:
    reader = csv.reader(csvfile)
    a = True
    points2= []
    for row in reader:
        if a:
            a=False
        else:
            points2.append([float(row[0]),float(row[1])])

pts = np.array(points)
pts2 = np.array(points2)
plt.scatter(pts[:,0],pts[:,1])
plt.scatter(pts2[:,0], pts2[:,1])
plt.legend(['New set','Old set'])
plt.title('Training set comparison')
plt.xlabel('x pos (m)')
plt.ylabel('y pos (m)')
plt.show()