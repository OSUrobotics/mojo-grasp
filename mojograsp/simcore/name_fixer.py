import os
import re


filepath = '/home/mothra/mojo-grasp/demos/rl_demo/data/JA_halfstate_A_rand/Test/Test/'
test_files = os.listdir(filepath)
num_cores = 16
for filename in test_files:
    points = re.findall('\d+',filename)
    if len(points) == 2:
        new_num = int(points[1]) * num_cores + int(points[0])
        os.rename(filepath+filename,filepath+'episode_'+str(new_num)+'.pkl')