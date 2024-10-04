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
    thing1 = (obj_rotation-reward_container['goal_orientation'])%(np.pi*2)
    thing2 = (reward_container['goal_orientation']-obj_rotation)%(np.pi*2)
    rot_temp = min(thing1,thing2)
    reward = -rot_temp*tholds['ROTATION_SCALING']
    return float(reward), False

def rotation(reward_container, tholds):
    # goal angle should be +/- pi
    # make the current angle set between +/- pi then subtract the two
    goal_dist = reward_container['distance_to_goal']/0.01 # divide to turn into cm
    obj_rotation = reward_container['object_orientation'][2]
    thing1 = (obj_rotation-reward_container['goal_orientation'])%(np.pi*2)
    thing2 = (reward_container['goal_orientation']-obj_rotation)%(np.pi*2)
    rot_temp = min(thing1,thing2)
    reward = -rot_temp*tholds['ROTATION_SCALING'] - goal_dist*tholds['DISTANCE_SCALING']
    return float(reward), False

def rotation_with_finger(reward_container, tholds):
    # goal angle should be +/- pi
    # make the current angle set between +/- pi then subtract the two
    goal_dist = reward_container['distance_to_goal']/0.01 # divide to turn into cm
    obj_rotation = reward_container['object_orientation'][2]
    thing1 = (obj_rotation-reward_container['goal_orientation'])%(np.pi*2)
    thing2 = (reward_container['goal_orientation']-obj_rotation)%(np.pi*2)
    rot_temp = min(thing1,thing2)
    ftemp = -max(reward_container['f1_dist'], reward_container['f2_dist']) * 100 
    reward = -rot_temp*tholds['ROTATION_SCALING'] - goal_dist*tholds['DISTANCE_SCALING']  + ftemp*tholds['CONTACT_SCALING']
    return float(reward), False

def slide_and_rotate(reward_container, tholds):
    obj_rotation = reward_container['object_orientation'][2]
    thing1 = (obj_rotation-reward_container['goal_orientation'])%(np.pi*2)
    thing2 = (reward_container['goal_orientation']-obj_rotation)%(np.pi*2)
    rotation_temp = min(thing1,thing2)
    ftemp = -max(reward_container['f1_dist'], reward_container['f2_dist']) * 100 # 100 here to make ftemp = -1 when at 1 cm
    temp = -reward_container['distance_to_goal']/reward_container['start_dist'] # should scale this so that it is -1 at start 
    ftemp,temp = max(ftemp,-2), max(temp, -2)
    tstep_reward = temp*tholds['DISTANCE_SCALING'] + ftemp*tholds['CONTACT_SCALING'] - rotation_temp/np.pi*tholds['ROTATION_SCALING']
    # print('reward start dist', reward_container['start_dist'])
    # print('things before scaling', temp, ftemp, -rotation_temp/np.pi)
    return float(tstep_reward), False

def contact_point(reward_container, tholds):
    if reward_container['timestep'] == 5:
        end_dist = np.abs(reward_container['goal_finger'] - reward_container['finger_pose'])
        start_dist = np.abs(reward_container['goal_finger'] - reward_container['start_finger'])
        dist_reward = -np.sum(end_dist-start_dist)
        return float(dist_reward/0.01-reward_container['distance_to_goal']/0.01), False
    else:
        return float(0), False

def direction(reward_container, tholds):
    return reward_container['dist_reward'] - 0.1*max(reward_container['f1_dist'],reward_container['f2_dist']), False


def triple_scaled_slide(reward_container, tholds):
    # print(reward_container)
    ftemp = -max(reward_container['f1_dist'], reward_container['f2_dist']) * 100 # 100 here to make ftemp = -1 when at 1 cm
    temp = -reward_container['distance_to_goal']/reward_container['start_dist'] # should scale this so that it is -1 at start 
    # print(reward_container['distance_to_goal'],reward_container['start_dist'], temp)
    obj_rotation = reward_container['object_orientation'][2]
    thing1 = (obj_rotation-reward_container['goal_orientation'])%(np.pi*2)
    thing2 = (reward_container['goal_orientation']-obj_rotation)%(np.pi*2)
    rotation_temp = -min(thing1,thing2)
    ftemp,temp = max(ftemp,-2), max(temp, -2)
    tstep_reward = temp*tholds['DISTANCE_SCALING'] + ftemp*tholds['CONTACT_SCALING'] + rotation_temp/np.pi*tholds['ROTATION_SCALING']
    return float(tstep_reward), False



"""
Begin HRL reward functions
"""
def manager(reward_container, tholds):
    obj_rotation = reward_container['object_orientation'][2]
    thing1 = (obj_rotation-reward_container['upper_goal_orientation'])%(np.pi*2)
    thing2 = (reward_container['upper_goal_orientation']-obj_rotation)%(np.pi*2)
    rotation_temp = -min(thing1,thing2)
    ftemp = -max(reward_container['f1_dist'], reward_container['f2_dist']) * 10 # 100 here to make ftemp = -1 when at 10 cm
    temp = -reward_container['upper_distance']/reward_container['start_dist'] # should scale this so that it is -1 at start 
    ftemp,temp = max(ftemp,-2), max(temp, -2)
    tstep_reward = temp*tholds['DISTANCE_SCALING'] + ftemp*tholds['CONTACT_SCALING'] + rotation_temp/np.pi*tholds['ROTATION_SCALING']
    # print(temp*tholds['DISTANCE_SCALING'],-rotation_temp*tholds['ROTATION_SCALING'],ftemp*tholds['CONTACT_SCALING'], tstep_reward)
    return tstep_reward, False

def manager_rotation(reward_container, tholds):
    obj_rotation = reward_container['object_orientation'][2]
    thing1 = (obj_rotation-reward_container['upper_goal_orientation'])%(np.pi*2)
    thing2 = (reward_container['upper_goal_orientation']-obj_rotation)%(np.pi*2)
    rotation_temp = -min(thing1,thing2)
    ftemp = -max(reward_container['f1_dist'], reward_container['f2_dist']) * 10 # 100 here to make ftemp = -1 when at 10 cm
    temp = -reward_container['upper_distance']/0.01 # should scale this so that it is -1 at start 
    # ftemp,temp = max(ftemp,-2), max(temp, -2)
    tstep_reward = temp*tholds['DISTANCE_SCALING'] + ftemp*tholds['CONTACT_SCALING'] + rotation_temp/np.pi*tholds['ROTATION_SCALING']
    # print(temp*tholds['DISTANCE_SCALING'],-rotation_temp*tholds['ROTATION_SCALING'],ftemp*tholds['CONTACT_SCALING'], tstep_reward)
    return tstep_reward, False

def manager_alt_1(reward_container, tholds):
    obj_rotation = reward_container['object_orientation'][2]
    thing1 = (obj_rotation-reward_container['upper_goal_orientation'])%(np.pi*2)
    thing2 = (reward_container['upper_goal_orientation']-obj_rotation)%(np.pi*2)
    rotation_temp = -min(thing1,thing2)
    ftemp = -max(reward_container['f1_dist'], reward_container['f2_dist']) * 10 # 100 here to make ftemp = -1 when at 10 cm
    temp = -reward_container['upper_distance']/reward_container['start_dist'] # should scale this so that it is -1 at start 
    ftemp,temp = max(ftemp,-2), max(temp, -2)
    tstep_reward = temp*tholds['DISTANCE_SCALING'] + ftemp*tholds['CONTACT_SCALING'] + rotation_temp/np.pi*tholds['ROTATION_SCALING']
    goal_vector = [reward_container['goal_position'][0]-reward_container['previous_pos'][0][0],
                       reward_container['goal_position'][1]-reward_container['previous_pos'][0][1]]
    movement_vector = [reward_container['object_pose'][0][0]-reward_container['previous_pos'][0][0],
                       reward_container['object_pose'][0][1]-reward_container['previous_pos'][0][1]]
    if np.isclose(np.linalg.norm(movement_vector),0):
        lower_reward_term = 0
    else:
        lower_reward_term =  np.dot(goal_vector,movement_vector)/(np.linalg.norm(goal_vector)*np.linalg.norm(movement_vector))
    # print(temp*tholds['DISTANCE_SCALING'],-rotation_temp*tholds['ROTATION_SCALING'],ftemp*tholds['CONTACT_SCALING'], tstep_reward)
    return tstep_reward + lower_reward_term*0.02, False

def worker_object_position(reward_container, tholds):
    goal_vector = [reward_container['goal_position'][0]-reward_container['previous_pos'][0][0],
                       reward_container['goal_position'][1]-reward_container['previous_pos'][0][1]]
    movement_vector = [reward_container['object_pose'][0][0]-reward_container['previous_pos'][0][0],
                       reward_container['object_pose'][0][1]-reward_container['previous_pos'][0][1]]
    obj_rotation = reward_container['object_orientation'][2]
    thing1 = (obj_rotation-reward_container['upper_goal_orientation'])%(np.pi*2)
    thing2 = (reward_container['upper_goal_orientation']-obj_rotation)%(np.pi*2)
    rotation_temp = -min(thing1,thing2)
    ftemp = -max(reward_container['f1_dist'], reward_container['f2_dist']) * 100 # 100 here to make ftemp = -1 when at 1 cm
    temp = -reward_container['upper_distance']/reward_container['start_dist'] # should scale this so that it is -1 at start 
    ftemp,temp = max(ftemp,-2), max(temp, -2)
    tstep_reward = temp*tholds['DISTANCE_SCALING'] + ftemp*tholds['CONTACT_SCALING'] + rotation_temp/np.pi*tholds['ROTATION_SCALING']
    if np.isclose(np.linalg.norm(movement_vector),0):
        lower_reward_term = 0
    else:
        lower_reward_term =  np.dot(goal_vector,movement_vector)/(np.linalg.norm(goal_vector)*np.linalg.norm(movement_vector))
    # print('goal, movement')
    # print(goal_vector, movement_vector)
    # print('dot, lower')
    # print(np.dot(goal_vector,movement_vector),lower_reward_term )
    return tstep_reward/2 + lower_reward_term/2, False

def worker_object_pose(reward_container, tholds):
    goal_vector = [reward_container['goal_position'][0]-reward_container['previous_pos'][0][0],
                       reward_container['goal_position'][1]-reward_container['previous_pos'][0][1],
                       reward_container['goal_orientation']-reward_container['previous_orientation'][2]]
    
    movement_vector = [reward_container['object_pose'][0][0]-reward_container['previous_pos'][0][0],
                       reward_container['object_pose'][0][1]-reward_container['previous_pos'][0][1],
                       reward_container['object_orientation'][2]-reward_container['previous_orientation'][2]]
    goal_vector[2] = goal_vector[2]/(-50/180*np.pi) * 0.08 
    movement_vector[2] = movement_vector[2]/(-50/180*np.pi) * 0.08 

    obj_rotation = reward_container['object_orientation'][2]
    thing1 = (obj_rotation-reward_container['upper_goal_orientation'])%(np.pi*2)
    thing2 = (reward_container['upper_goal_orientation']-obj_rotation)%(np.pi*2)
    rotation_temp = -min(thing1,thing2)
    ftemp = -max(reward_container['f1_dist'], reward_container['f2_dist']) * 100 # 100 here to make ftemp = -1 when at 1 cm
    temp = -reward_container['upper_distance']/reward_container['start_dist'] # should scale this so that it is -1 at start 
    ftemp,temp = max(ftemp,-2), max(temp, -2)
    tstep_reward = temp*tholds['DISTANCE_SCALING'] + ftemp*tholds['CONTACT_SCALING'] + rotation_temp/np.pi*tholds['ROTATION_SCALING']
    if np.isclose(np.linalg.norm(movement_vector),0,atol=0.0001):
        lower_reward_term = 0
    else:
        lower_reward_term =  np.dot(goal_vector,movement_vector)/(np.linalg.norm(goal_vector)*np.linalg.norm(movement_vector))
    # print('goal, movement')
    # print(goal_vector, movement_vector)
    # print('dot, lower')
    # print(np.dot(goal_vector,movement_vector),lower_reward_term )
    # may need to get this 
    return tstep_reward/2 + lower_reward_term/2, False

def worker_object_pose_finger(reward_container, tholds):
    goal_vector = [reward_container['goal_position'][0]-reward_container['previous_pos'][0][0],
                       reward_container['goal_position'][1]-reward_container['previous_pos'][0][1],
                       reward_container['goal_orientation']-reward_container['previous_orientation'][2],
                       reward_container['goal_finger'][0] - (reward_container['previous_finger_pose'][0]-reward_container['previous_finger_pose'][2]),
                       reward_container['goal_finger'][1] - (reward_container['previous_finger_pose'][1]-reward_container['previous_finger_pose'][3])]
    
    movement_vector = [reward_container['object_pose'][0][0]-reward_container['previous_pos'][0][0],
                       reward_container['object_pose'][0][1]-reward_container['previous_pos'][0][1],
                       reward_container['object_orientation'][2]-reward_container['previous_orientation'][2],
                       (reward_container['finger_pose'][0]-reward_container['previous_finger_pose'][2]) - (reward_container['previous_finger_pose'][0]-reward_container['previous_finger_pose'][2]),
                       (reward_container['finger_pose'][1]-reward_container['previous_finger_pose'][3]) - (reward_container['previous_finger_pose'][1]-reward_container['previous_finger_pose'][3])]
    goal_vector[2] = goal_vector[2] /(-50/180*np.pi) * 0.08 
    movement_vector[2] = movement_vector[2] /(-50/180*np.pi) * 0.08 
    goal_vector[3] = goal_vector[3] * 2
    goal_vector[4] = goal_vector[4] * 2
    movement_vector[3] = movement_vector[3] * 2
    movement_vector[4] = movement_vector[4] * 2
    obj_rotation = reward_container['object_orientation'][2]
    thing1 = (obj_rotation-reward_container['upper_goal_orientation'])%(np.pi*2)
    thing2 = (reward_container['upper_goal_orientation']-obj_rotation)%(np.pi*2)
    rotation_temp = -min(thing1,thing2)
    ftemp = -max(reward_container['f1_dist'], reward_container['f2_dist']) * 100 # 100 here to make ftemp = -1 when at 1 cm
    temp = -reward_container['upper_distance']/reward_container['start_dist'] # should scale this so that it is -1 at start 
    ftemp,temp = max(ftemp,-2), max(temp, -2)
    tstep_reward = temp*tholds['DISTANCE_SCALING'] + ftemp*tholds['CONTACT_SCALING'] + rotation_temp/np.pi*tholds['ROTATION_SCALING']
    if np.isclose(np.linalg.norm(movement_vector),0,atol=0.0001):
        lower_reward_term = 0
    else:
        lower_reward_term =  np.dot(goal_vector,movement_vector)/(np.linalg.norm(goal_vector)*np.linalg.norm(movement_vector))

    if lower_reward_term > 0.8:
        # separated = np.array([goal_vector[i] * movement_vector[i] for i in range(len(goal_vector))])
        print('reward term:',lower_reward_term)
        print('goal vector:', goal_vector)
        print('move vector:', movement_vector)
    return tstep_reward/2 + lower_reward_term/2, False

def worker_object_pose_finger_rotation(reward_container, tholds):
    goal_vector = [reward_container['goal_position'][0]-reward_container['previous_pos'][0][0],
                       reward_container['goal_position'][1]-reward_container['previous_pos'][0][1],
                       reward_container['goal_orientation']-reward_container['previous_orientation'][2],
                       reward_container['goal_finger'][0] - (reward_container['previous_finger_pose'][0]-reward_container['previous_finger_pose'][2]),
                       reward_container['goal_finger'][1] - (reward_container['previous_finger_pose'][1]-reward_container['previous_finger_pose'][3])]
    
    movement_vector = [reward_container['object_pose'][0][0]-reward_container['previous_pos'][0][0],
                       reward_container['object_pose'][0][1]-reward_container['previous_pos'][0][1],
                       reward_container['object_orientation'][2]-reward_container['previous_orientation'][2],
                       (reward_container['finger_pose'][0]-reward_container['previous_finger_pose'][2]) - (reward_container['previous_finger_pose'][0]-reward_container['previous_finger_pose'][2]),
                       (reward_container['finger_pose'][1]-reward_container['previous_finger_pose'][3]) - (reward_container['previous_finger_pose'][1]-reward_container['previous_finger_pose'][3])]
    goal_vector[2] = goal_vector[2] /(-50/180*np.pi) * 0.08 
    movement_vector[2] = movement_vector[2] /(-50/180*np.pi) * 0.08 
    goal_vector[3] = goal_vector[3] * 2
    goal_vector[4] = goal_vector[4] * 2
    movement_vector[3] = movement_vector[3] * 2
    movement_vector[4] = movement_vector[4] * 2
    obj_rotation = reward_container['object_orientation'][2]
    thing1 = (obj_rotation-reward_container['upper_goal_orientation'])%(np.pi*2)
    thing2 = (reward_container['upper_goal_orientation']-obj_rotation)%(np.pi*2)
    rotation_temp = -min(thing1,thing2)
    ftemp = -max(reward_container['f1_dist'], reward_container['f2_dist']) * 100 # 100 here to make ftemp = -1 when at 1 cm
    temp = -reward_container['upper_distance']/0.01# should scale this so that it is -1 at start 
    ftemp,temp = max(ftemp,-2), max(temp, -2)
    tstep_reward = temp*tholds['DISTANCE_SCALING'] + ftemp*tholds['CONTACT_SCALING'] + rotation_temp/np.pi*tholds['ROTATION_SCALING']
    if np.isclose(np.linalg.norm(movement_vector),0,atol=0.0001):
        lower_reward_term = 0
    else:
        lower_reward_term =  np.dot(goal_vector,movement_vector)/(np.linalg.norm(goal_vector)*np.linalg.norm(movement_vector))

    separated = np.array([goal_vector[i] * movement_vector[i] for i in range(len(goal_vector))])
    return lower_reward_term/2, False

def sparse_multigoal(reward_container, tholds):
    reward = reward_container['goals_reached'] * tholds['SUCCESS_REWARD']*10
    # if reward==0:
    #     reward = -1
    return reward, False

def worker_multigoal(reward_container,tholds):
    goal_vector = [reward_container['goal_position'][0]-reward_container['previous_pos'][0][0],
                       reward_container['goal_position'][1]-reward_container['previous_pos'][0][1],
                       reward_container['goal_orientation']-reward_container['previous_orientation'][2]]
    
    movement_vector = [reward_container['object_pose'][0][0]-reward_container['previous_pos'][0][0],
                       reward_container['object_pose'][0][1]-reward_container['previous_pos'][0][1],
                       reward_container['object_orientation'][2]-reward_container['previous_orientation'][2]]
    goal_vector[2] = goal_vector[2]/(-50/180*np.pi) * 0.08 
    movement_vector[2] = movement_vector[2]/(-50/180*np.pi) * 0.08 
    if np.isclose(np.linalg.norm(movement_vector),0,atol=0.0001):
        lower_reward_term = 0
    else:
        lower_reward_term =  np.dot(goal_vector,movement_vector)/(np.linalg.norm(goal_vector)*np.linalg.norm(movement_vector))
    upper_reward_term = reward_container['goals_reached'] * tholds['SUCCESS_REWARD']
    if upper_reward_term==0:
        upper_reward_term = -1
    return upper_reward_term*0+lower_reward_term, False
    

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # goal_angs = np.linspace(-12.56,12.56,1000)
    # end_angs = np.linspace(-12.56,12.56,1000)
    goal = 1
    finger_distances = np.linspace(0,0.1,1000)
    tholds = {'DISTANCE_SCALING':0.1, 'CONTACT_SCALING':0.2, 'ROTATION_SCALING': 1}
    y = np.zeros(len(finger_distances))
    for i, a1 in enumerate(finger_distances):
        reward_container = {'distance_to_goal':0, 'start_dist':1,'f1_dist':a1, 'f2_dist':0, 'object_orientation':[0,0,0], 'goal_orientation':0}
        y[i],_ = triple_scaled_slide(reward_container,tholds)

    plt.plot(finger_distances,y)
    # plt.title(f'reward for goal angle of {goal}')
    plt.xlabel('finger_error')
    plt.ylabel('reward')
    plt.show()

    # x,y = np.meshgrid(goal_angs,end_angs)
    # z = np.zeros((len(goal_angs),len(end_angs)))
    # tholds = {'DISTANCE_SCALING':0.1, 'CONTACT_SCALING':0.2}
    # for i,a1 in enumerate(goal_angs):
    #     for j,a2 in enumerate(end_angs):
    #         reward_container = {'distance_to_goal':0, 'f1_dist':0, 'f2_dist':0, 'object_orientation':[0,0,a2], 'goal_orientation':a1}
    #         r,_ = rotation_with_finger(reward_container, tholds)
    #         z[i,j] = r
    # fig,ax = plt.subplots(subplot_kw={"projection": "3d"})
    # ax.scatter(x, y, z, linewidth=0, antialiased=False)
    # plt.xlabel('goal angs')
    # plt.ylabel('ending angs')
    # plt.show()