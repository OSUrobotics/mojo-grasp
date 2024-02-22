# reward functions
import numpy as np
import scipy.spatial.transform.rotation as R



def sparse(reward_container, tholds):
    tstep_reward = -1 + 2*(reward_container['distance_to_goal'] < tholds['SUCCESS_THRESHOLD'])

    return float(tstep_reward), False

def distance(reward_container, tholds):
    tstep_reward = max(-reward_container['distance_to_goal'],-1)
    return float(tstep_reward), False

def distance_finger(reward_container, tholds):
    tstep_reward = max(-reward_container['distance_to_goal']*tholds['DISTANCE_SCALING']- max(reward_container['f1_dist'],reward_container['f2_dist'])*tholds['CONTACT_SCALING'],-1)
    return float(tstep_reward), False

def hinge_distance(reward_container, tholds):
    tstep_reward = reward_container['distance_to_goal'] < tholds['SUCCESS_THRESHOLD'] + max(-reward_container['distance_to_goal'] - max(reward_container['f1_dist'],reward_container['f2_dist'])*tholds['CONTACT_SCALING'],-1)
    return float(tstep_reward), False

def slope(reward_container, tholds):
    tstep_reward = reward_container['slope_to_goal'] * tholds['DISTANCE_SCALING']
    return float(tstep_reward), False

def slope_finger(reward_container, tholds):
    tstep_reward = max(reward_container['slope_to_goal'] * tholds['DISTANCE_SCALING']  - max(reward_container['f1_dist'],reward_container['f2_dist'])*tholds['CONTACT_SCALING'],-1)
    return float(tstep_reward), False

def smart(reward_container, tholds):
    ftemp = max(reward_container['f1_dist'],reward_container['f2_dist'])
    temp = -reward_container['distance_to_goal'] * (1 + 4*reward_container['plane_side'])
    tstep_reward = max(temp*tholds['DISTANCE_SCALING'] - ftemp*tholds['CONTACT_SCALING'],-1)
    return float(tstep_reward), False

def scaled(reward_container, tholds):
    ftemp = max(reward_container['f1_dist'], reward_container['f2_dist']) * 100 # 100 here to make ftemp = -1 when at 1 cm
    temp = -reward_container['distance_to_goal']/reward_container['start_dist'] * (1 + 4*reward_container['plane_side'])
    tstep_reward = temp*tholds['DISTANCE_SCALING'] - ftemp*tholds['CONTACT_SCALING']
    return float(tstep_reward), False

def double_scaled(reward_container, tholds):
    ftemp = -max(reward_container['f1_dist'], reward_container['f2_dist']) * 100 # 100 here to make ftemp = -1 when at 1 cm
    temp = -reward_container['distance_to_goal']/reward_container['start_dist'] # should scale this so that it is -1 at start 
    ftemp,temp = max(ftemp,-2), max(temp, -2)
    tstep_reward = temp*tholds['DISTANCE_SCALING'] + ftemp*tholds['CONTACT_SCALING']
    return float(tstep_reward), False

def sfs(reward_container, tholds):
    tstep_reward = reward_container['slope_to_goal'] * tholds['DISTANCE_SCALING'] - max(reward_container['f1_dist'],reward_container['f2_dist'])*tholds['CONTACT_SCALING']
    if (reward_container['distance_to_goal'] < tholds['SUCCESS_THRESHOLD']) & (np.linalg.norm(reward_container['object_velocity']) <= 0.05):
        tstep_reward += tholds['SUCCESS_REWARD']
        done2 = True
    return float(tstep_reward), done2

def dfs(reward_container, tholds):
    ftemp = max(reward_container['f1_dist'],reward_container['f2_dist'])
    # assert ftemp >= 0
    tstep_reward = -reward_container['distance_to_goal'] * tholds['DISTANCE_SCALING']  - ftemp*tholds['CONTACT_SCALING']
    if (reward_container['distance_to_goal'] < tholds['SUCCESS_THRESHOLD']) & (np.linalg.norm(reward_container['object_velocity']) <= 0.05):
        tstep_reward += tholds['SUCCESS_REWARD']
        done2 = True
    return float(tstep_reward), done2

def double_smart(reward_container, tholds):
    ftemp = max(reward_container['f1_dist'],reward_container['f2_dist'])
    if ftemp > 0.001:
        ftemp = ftemp*ftemp*1000
    temp = -reward_container['distance_to_goal'] * (1 + 4*reward_container['plane_side'])
    tstep_reward = max(temp*tholds['DISTANCE_SCALING'] - ftemp*tholds['CONTACT_SCALING'],-1)
    return float(tstep_reward), False

def multi_scaled(reward_container, tholds):
    ftemp = -max(reward_container['f1_dist'], reward_container['f2_dist']) * 100 # 100 here to make ftemp = -1 when at 1 cm
    temp = -reward_container['distance_to_goal']/reward_container['start_dist'] # should scale this so that it is -1 at start 
    ftemp,temp = max(ftemp,-2), max(temp, -2)
    success = reward_container['distance_to_goal'] < tholds['SUCCESS_THRESHOLD']
    # print(reward_container['distance_to_goal'], self.SUCCESS_THRESHOLD)
    done2 = success
    if done2:
        print(f'dist:{temp*tholds["DISTANCE_SCALING"]}, contact:{ftemp*tholds["CONTACT_SCALING"]}, success:{success * tholds["SUCCESS_REWARD"]}, start {reward_container["start_dist"]}')
    tstep_reward = temp*tholds['DISTANCE_SCALING'] + ftemp*tholds['CONTACT_SCALING'] + success * tholds['SUCCESS_REWARD']
    return float(tstep_reward), False


def solo_rotation(reward_container, tholds):
    # goal angle should be +/- pi
    # make the current angle set between +/- pi then subtract the two

    obj_rotation = reward_container['object_orientation'][2]
    obj_rotation = (obj_rotation + np.pi)%(np.pi*2)
    obj_rotation = obj_rotation - np.pi
    reward = -abs(reward_container['goal_orientation'] - obj_rotation)
    return float(reward), False


def rotation(reward_container, tholds):
    # goal angle should be +/- pi
    # make the current angle set between +/- pi then subtract the two
    goal_dist = reward_container['distance_to_goal']/0.01 # divide to turn into cm
    obj_rotation = reward_container['object_orientation'][2]
    obj_rotation = (obj_rotation + np.pi)%(np.pi*2)
    obj_rotation = obj_rotation - np.pi
    reward = -abs(reward_container['goal_orientation'] - obj_rotation) - goal_dist*tholds['DISTANCE_SCALING']
    return float(reward), False


def rotation_with_finger(reward_container, tholds):
    # goal angle should be +/- pi
    # make the current angle set between +/- pi then subtract the two
    goal_dist = reward_container['distance_to_goal']/0.01 # divide to turn into cm
    obj_rotation = reward_container['object_orientation'][2]
    obj_rotation = (obj_rotation + np.pi)%(np.pi*2)
    obj_rotation = obj_rotation - np.pi
    ftemp = -max(reward_container['f1_dist'], reward_container['f2_dist']) * 100 
    reward = -abs(reward_container['goal_orientation'] - obj_rotation) - goal_dist*tholds['DISTANCE_SCALING']  + ftemp*tholds['CONTACT_SCALING']
    return float(reward), False

def slide_and_rotate(reward_container, tholds):
    obj_rotation = reward_container['object_orientation'][2]
    obj_rotation = (obj_rotation + np.pi)%(np.pi*2)
    obj_rotation = obj_rotation - np.pi
    rotation_temp = -abs(reward_container['goal_orientation'] - obj_rotation)/np.pi # divide by pi to make rotation reward -1 when we are at opposite side
    ftemp = -max(reward_container['f1_dist'], reward_container['f2_dist']) * 100 # 100 here to make ftemp = -1 when at 1 cm
    temp = -reward_container['distance_to_goal']/reward_container['start_dist'] # should scale this so that it is -1 at start 
    ftemp,temp = max(ftemp,-2), max(temp, -2)
    tstep_reward = temp*tholds['DISTANCE_SCALING'] + ftemp*tholds['CONTACT_SCALING'] + rotation_temp
    return float(tstep_reward), False