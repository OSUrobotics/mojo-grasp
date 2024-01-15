import os

JAs = ['Full', 'Half']
hand_params = ['Hand', 'NoHand']
Action_space = ['FTP','JA']
hands = ['BothInterp','PalmExtrap','FingerExtrap']


for k1 in JAs:
    for k2 in hand_params:
        for k3 in Action_space:
            for k4 in hands:
                temp = '_'.join([k1,k2,k3,k4])
                os.mkdir(temp)