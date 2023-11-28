import os
import re


filepath = '/home/orochi/mojo/mojo-grasp/demos/rl_demo/data/hand_B_baseline_JA/Test/'
test_files = os.listdir(filepath)
num_cores = 16
# print(test_files)
for filename in test_files:
    points = re.findall('\d+',filename)
    # print(points)
    new_num = int(points[1]) * num_cores + int(points[0])
    os.rename(filepath+filename,filepath+'episode_'+str(new_num)+'.pkl')