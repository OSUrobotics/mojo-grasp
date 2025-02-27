import sys
sys.path.append('/home/mothra/pybullet-planning')
# import pybullet_tools.utils as pp

from demos.rl_demo.learnpettingzooHRL import *
from scipy.interpolate import interp2d
import torch 
from stable_baselines3.common.buffers import ReplayBuffer, RolloutBuffer
from demos.rl_demo.policies import PPOExpertData
from copy import deepcopy
import re
from multiprocessing import pool
from stable_baselines3.common.utils import obs_as_tensor
import torch as th

def make_env(arg_dict=None,rank=0,hand_info=None,previous_policy=None):
    def _init():
        import pybullet as p1
        env, _, _ = make_pybullet(arg_dict, p1, rank, hand_info)
        env = PettingZooAECWrapper(env)
        load_path = arg_dict['load_path']
        partner = StaticPolicyAgent(previous_policy.policy)
        env.add_partner_agent(partner,player_num=1)
        return env
    return _init

def collect_expert(env,poses):
    env.evaluate()
    # start small
    
    pose_orders = []
    for i in range(int(len(poses[0]))-4):
        goal_poses = poses[0][i].tolist()
        # print(goal_poses)
        goal_poses = np.reshape(goal_poses,(2,2)).tolist()
        start_point = np.array([0,0])
        order = [start_point.tolist()]
        for _ in range(2):
            dists = np.linalg.norm(start_point - np.array(goal_poses),axis=1)
            # print(dists, start_point, np.array(goal_poses))
            next_ind = np.argmin(dists)
            order.append(goal_poses[next_ind])
            start_point = np.array(goal_poses[next_ind])
            goal_poses.pop(next_ind)
        pose_orders.append(order)

    buffer = RolloutBuffer(40*len(pose_orders),env.observation_space('manager'),env.action_space('manager'))
    just_goals = {'goals':[],'actions':[], 'start':[]}
    for order in pose_orders:
        # go through each of the orders we made earlier and set the simulator to those points
        # using 5 steps per spot as the goal point, starting the object at the desired point
        # and interpolating from that spot
        env.reset({'start_pos':order[0],'finger_angs':[-np.pi/2,0,np.pi/2,0]})
        goals=[order[int(i/20)+1] for i in range(40)]
        order_array = np.array(order)
        object_poses = [np.linspace(order_array[i],order_array[i+1],20) for i in range(len(order_array)-1)]
        object_poses = np.array(object_poses)
        # print(object_poses)
        object_poses = object_poses.reshape(40,2)
        print('about to get episode')
        done = False
        i=0
        while not done:
            env.step(np.array(goals[i]))
            env.step(np.array([0,0,0,0]),direct_control=[[object_poses[i][0],object_poses[i][1]+0.1],[0,0,0,1]])
            observation = env.observations['manager']
            action = (np.array(goals[i]) - env.manager_normalizer['mins'])/env.manager_normalizer['diff']*2 -1
            # print(np.shape(observation))
            assert all(abs(action)< 1)  
            
            # normalized_action = (action+1) * self.manager_normalizer['diff']/2 + self.manager_normalizer['mins']
            reward = env.rewards['manager']
            log_probs = torch.zeros(1)
            values=torch.zeros(1)
            just_goals['actions'].append(action)
            just_goals['goals'].append(goals)
            just_goals['start'].append(order[0])
            buffer.add(observation,action,reward,i==0,values,log_probs)
            done = env.manipulation_phase.exit_condition()
            i+=1
    env.train()
    return buffer, just_goals

def pool_data_extraction(episode_file, build_state, key_list, reward_function, weights):
    '''
    This takes a given episode file, a build state function and a set of keys and returns a list
    of all the data from that episode processed into data that can be fed directly into the network.
    Designed to be used with Pool.starmap on a folder 
    full of data
    '''
    with open(episode_file, 'rb') as ef:
        tempdata = pkl.load(ef)
    data = tempdata['timestep_list']
    next_state = []
    state = []
    action = []
    reward = []
    done = []
    start_state = tempdata['start_state']
    for i in range(len(data)):
        data[len(data)-1-i]['state']['previous_state'] = []
        for j in range(4):
            if len(data)-1-i-4+j < 0:
                print('too soon, just using the first one')
                data[len(data)-1-i]['state']['previous_state'].append(deepcopy(start_state))
            else:
                data[len(data)-1-i]['state']['previous_state'].append(deepcopy(data[len(data)-1-i-4+j]['state']))
        next_state.append(build_state(data[i]['state'], key_list))
        if i != 0:
            state.append(build_state(data[i]['state'], key_list))
        reward.append(reward_function(data[i]['reward'], weights))
        action.append(data[i]['action']['actor_output'])
        done.append(i==0)
    state_1 = deepcopy(start_state)
    state_1['previous_state'] = []
    for i in range(4):
        state_1['previous_state'].append(deepcopy(start_state))
    state.append(build_state(state_1, key_list))
    state.reverse()
    next_state.reverse()
    action.reverse()
    reward.reverse()
    done.reverse()
    return {'observation':state,'action':action,'reward':reward,'next_state':next_state,'done':done}

def load_expert_translation(folder, build_state, key_list, reward_function, reward_weights, n_rollout_steps) -> bool:
    # Make a list of offline observations, actions and trees
    print("INFO: Making offline rollouts")
    n_steps = 0
    # TODO: Do we need callbacks in offline rollouts?
    # callback.update_locals(locals())
    # callback.on_rollout_start()observation_space, action_space,

    episode_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith('.pkl')]
    filenames_only = [f for f in os.listdir(folder) if f.lower().endswith('.pkl')]
    
    filenums = [re.findall('\d+',f) for f in filenames_only]
    final_filenums = []
    for i in filenums:
        if len(i) > 0 :
            final_filenums.append(int(i[0]))
    
    sorted_inds = np.argsort(final_filenums)
    final_filenums = np.array(final_filenums)
    temp = final_filenums[sorted_inds]
    episode_files = np.array(episode_files)
    filenames_only = np.array(filenames_only)

    episode_files = episode_files[sorted_inds].tolist()
    thing = [[ef, build_state, key_list, reward_function, reward_weights] for ef in episode_files]
    print('applying async')
    data_list = pool.starmap(pool_data_extraction,thing)
    data_dict = {'observation':[],'action':[],'reward':[],'next_state':[],'done':[]}
    # Sample expert episode
    for episode_dict in data_list:
        data_dict['observation'].extend(episode_dict['observation'])
        data_dict['action'].extend(episode_dict['action'])
        data_dict['reward'].extend(episode_dict['reward'])
        data_dict['next_state'].extend(episode_dict['next_state'])
        data_dict['done'].extend(episode_dict['done'])
    
    data_dict['observation'] = np.array(data_dict['observation'])

    data_dict['action'] = np.array(data_dict['observation'])
    data_dict['reward'] = np.array(data_dict['reward'])
    data_dict['next_state'] = np.array(data_dict['next_state'])
    data_dict['done'] = np.array(data_dict['done'])
    new_inds = np.random.shuffle(list(range(len(data_dict['observation']))))
    data_dict['observation'] = data_dict['observation'][new_inds]
    data_dict['action'] = data_dict['action'][new_inds]
    data_dict['reward'] = data_dict['reward'][new_inds]
    data_dict['next_state'] = data_dict['next_state'][new_inds]
    data_dict['done'] = data_dict['done'][new_inds]
    expert_buffer = RolloutBuffer(n_rollout_steps,observation_space,action_space)
    while n_steps < n_rollout_steps:
        with th.no_grad():
            # Convert to pytorch tensor or to TensorDict
            obs_tensor = obs_as_tensor(data_dict['observation'][n_steps], 'cuda')
            actions = obs_as_tensor(data_dict['action'][n_steps], 'cuda')
            next_state = th.tensor(data_dict['next_state'][n_steps], dtype=th.float32, device='cuda')
            actions, values, log_probs = self.policy.forward_expert(obs_tensor, actions)
            rewards = data_dict['reward']

        actions = obs_as_tensor(data_dict['reward'][n_steps], 'cuda')

        expert_buffer.add(
            obs_tensor,
            actions,
            rewards,
            next_state,
            values,
            log_probs,
        )

        self._last_episode_starts_expert = dones

    next_obs = self._flatten_obs(batch['next_observation'],
                                    self.observation_space)  # Get the next observation to calculate the values

    with th.no_grad():
        # Compute value for the last timestep
        episode_starts = th.tensor(dones, dtype=th.float32, device=self.device)
        values = self.policy.predict_values(
            obs_as_tensor(next_obs, self.device))  # pylint: disable=unexpected-keyword-arg
    expert_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

    if self.verbose > 0:
        print("INFO: Finished making offline rollouts")
    # callback.on_rollout_end()
    # callback.update_locals(locals())
    return True


def train_expert(filepath, learn_type='run', num_cpu=16):
    # Create the vectorized environment
    print('cuda y/n?', get_device())

    with open(filepath, 'r') as argfile:
        args = json.load(argfile)

    key_file = os.path.abspath(__file__)
    key_file = os.path.dirname(key_file)
    key_file = os.path.join(key_file,'resources','hand_bank','hand_params.json')
    with open(key_file,'r') as hand_file:
        hand_params = json.load(hand_file)


    import pybullet as pybullet_instance
    # from pettingzoo.utils.conversions import aec_to_parallel 
    from pantheonrl.envs.pettingzoo import PettingZooAECWrapper
    from pantheonrl.common.agents import OnPolicyAgent
    env, _ , poses= make_pybullet(args, pybullet_instance, [0,1], hand_params, viz=False)
    # env = pettingzoowrapper.WrapWrap(env)
    # env = ss.pad_action_space_v0(env)
        
    if args['model']== "PPO":
        model_type = PPO
    elif args['model'] == "PPO_Expert":
        model_type = PPOExpertData
        expert_buffer,_ = collect_expert(env,poses)
    env = PettingZooAECWrapper(env)
    if args['load_path'] == '//':
        partner = OnPolicyAgent(PPO('MlpPolicy', env.getDummyEnv(1), verbose=1, tensorboard_log=args['tname']+'/worker'), tensorboard_log=args['tname']+'/worker')
        env.add_partner_agent(partner, player_num=1)
        train_timesteps = int(args['evaluate']*(args['tsteps']+1))
        worker_callback = pettingzoowrapper.WorkerEvaluateCallback(partner.model, args['save_path'])
        callback = pettingzoowrapper.ZooEvaluateCallback(env,n_eval_episodes=int(1200), eval_freq=int(train_timesteps), best_model_save_path=args['save_path'],callback_on_new_best=worker_callback)

    else:
        if args['worker_frozen']:
            load_path = args['load_path']
            previous_policy = PPO("MlpPolicy", None, _init_setup_model=False,device='cuda').load(load_path + '/best_model.zip',device='cuda') 
            partner = StaticPolicyAgent(previous_policy.policy)
            env.add_partner_agent(partner, player_num=1)
            train_timesteps = int(args['evaluate']*(args['tsteps']+1))
            # worker_callback = pettingzoowrapper.WorkerEvaluateCallback(partner.policy, args['save_path'])
            callback = pettingzoowrapper.ZooEvaluateCallback(env,n_eval_episodes=int(1200), eval_freq=int(train_timesteps), best_model_save_path=args['save_path'])
        else:
            load_path = args['load_path']
            previous_policy = PPO("MlpPolicy", None, _init_setup_model=False,device='cuda').load(load_path + '/best_model.zip',device='cuda') 
            partner = OnPolicyAgent(previous_policy, tensorboard_log=args['tname']+'/worker')
            env.add_partner_agent(partner, player_num=1)
            train_timesteps = int(args['evaluate']*(args['tsteps']+1))
            worker_callback = pettingzoowrapper.WorkerEvaluateCallback(partner.model, args['save_path'])
            callback = pettingzoowrapper.ZooEvaluateCallback(env,n_eval_episodes=int(1200), eval_freq=int(train_timesteps), best_model_save_path=args['save_path'],callback_on_new_best=worker_callback)

    # The second parameter ensures that the partner is assigned to a certain
    # player number. Forgetting this parameter would mean that all of the
    # partner agents can be picked as `player 2`, but none of them can be
    # picked as `player 3`. 
    # env.add_partner_agent(partner, player_num=1)
    # train_timesteps = int(args['evaluate']*(args['tsteps']+1))
    # worker_callback = pettingzoowrapper.WorkerEvaluateCallback(partner.model, args['save_path'])
    # callback = pettingzoowrapper.ZooEvaluateCallback(env,n_eval_episodes=int(1200), eval_freq=int(train_timesteps), best_model_save_path=args['save_path'],callback_on_new_best=worker_callback)

    if args['model']== "PPO":
        if 'mims' in args['manager_state_list']:
            model = model_type('CnnPolicy', env, tensorboard_log=args['tname']+'/manager')
        else:
            model = model_type("MlpPolicy", env, tensorboard_log=args['tname']+'/manager')
    elif args['model'] == "PPO_Expert":
        if 'mims' in args['manager_state_list']:
            model = model_type('CnnPolicy', env, tensorboard_log=args['tname']+'/manager', expert_buffer=expert_buffer)
        else:
            model = model_type("MlpPolicy", env, tensorboard_log=args['tname']+'/manager', expert_buffer=expert_buffer)
    try:
        model.learn(total_timesteps=args['epochs']*(args['tsteps']+1), callback=callback)
        filename = os.path.dirname(filepath)
        model.save(filename+'/manager_last_model')
        partner.model.save(filename+'/worker_last_model')
        merge_from_folder(args['save_path']+'Test/')
        # multiprocess_evaluate(model,vec_env)
    except KeyboardInterrupt:
        filename = os.path.dirname(filepath)
        model.save(filename+'/manager_canceled_model')
        partner.model.save(filename+'/worker_canceled_model')

def train_expert_multiprocess(filepath, learn_type='run', num_cpu=16):
    # Create the vectorized environment
    print('cuda y/n?', get_device())

    with open(filepath, 'r') as argfile:
        args = json.load(argfile)

    key_file = os.path.abspath(__file__)
    key_file = os.path.dirname(key_file)
    key_file = os.path.join(key_file,'resources','hand_bank','hand_params.json')
    with open(key_file,'r') as hand_file:
        hand_params = json.load(hand_file)


    # from pettingzoo.utils.conversions import aec_to_parallel 
    from pantheonrl.envs.pettingzoo import PettingZooAECWrapper
    from pantheonrl.common.agents import OnPolicyAgent
    # env = pettingzoowrapper.WrapWrap(env)
    # env = ss.pad_action_space_v0(env)
        
    pose_list, eval_pose_list, _, _, _, _, _ = load_set(args) 
    poses = [pose_list,eval_pose_list]
    if args['load_path'] == '//':
        raise KeyError("need to load the low level policy to multiprocess")
    else:
        load_path = args['load_path']
        p_p = PPO("MlpPolicy", None, _init_setup_model=False,device='cpu').load(load_path + '/best_model.zip',device='cpu') 
        env = SubprocVecEnv([make_env(args, [i,num_cpu], hand_params,p_p) for i in range(num_cpu)])
        if args['model']== "PPO":
            model_type = PPO
        elif args['model'] == "PPO_Expert":
            model_type = PPOExpertData
            expert_buffer,_ = collect_expert(env,poses)
        

        # env.add_partner_agent(partner, player_num=1)
        train_timesteps = int(args['evaluate']*(args['tsteps']+1))
        # worker_callback = pettingzoowrapper.WorkerEvaluateCallback(partner.policy, args['save_path'])
        callback = pettingzoowrapper.ZooEvaluateCallback(env,n_eval_episodes=int(1200), eval_freq=int(train_timesteps), best_model_save_path=args['save_path'])
    
    if args['model']== "PPO":
        if 'mims' in args['manager_state_list']:
            model = model_type('CnnPolicy', env, tensorboard_log=args['tname']+'/manager')
            assert 1==0
        else:
            model = model_type("MlpPolicy", env, tensorboard_log=args['tname']+'/manager')
    elif args['model'] == "PPO_Expert":
        if 'mims' in args['manager_state_list']:
            model = model_type('CnnPolicy', env, tensorboard_log=args['tname']+'/manager', expert_buffer=expert_buffer)
        else:
            model = model_type("MlpPolicy", env, tensorboard_log=args['tname']+'/manager', expert_buffer=expert_buffer)
    try:
        model.learn(total_timesteps=args['epochs']*(args['tsteps']+1), callback=callback)
        filename = os.path.dirname(filepath)
        model.save(filename+'/manager_last_model')
        merge_from_folder(args['save_path']+'Test/')
        # multiprocess_evaluate(model,vec_env)
    except KeyboardInterrupt:
        filename = os.path.dirname(filepath)
        model.save(filename+'/manager_canceled_model')

def test_expert_data(filepath):
    print('checking to see how good expert really is')
    with open(filepath, 'r') as argfile:
        args = json.load(argfile)

    key_file = os.path.abspath(__file__)
    key_file = os.path.dirname(key_file)
    key_file = os.path.join(key_file,'resources','hand_bank','hand_params.json')
    with open(key_file,'r') as hand_file:
        hand_params = json.load(hand_file)


    import pybullet as pybullet_instance
    # from pettingzoo.utils.conversions import aec_to_parallel 
    from pantheonrl.envs.pettingzoo import PettingZooAECWrapper
    from pantheonrl.common.agents import OnPolicyAgent
    env, _ , poses= make_pybullet(args, pybullet_instance, [0,1], hand_params, viz=True)
    # env = pettingzoowrapper.WrapWrap(env)
    # env = ss.pad_action_space_v0(env)
        
    if args['model']== "PPO":
        model_type = PPO
    elif args['model'] == "PPO_Expert":
        model_type = PPOExpertData
        _, expert_examples = collect_expert(env,poses)
    env = PettingZooAECWrapper(env)
    if args['load_path'] == '//':
        assert False
    else:
        load_path = args['load_path']
        previous_policy = PPO("MlpPolicy", None, _init_setup_model=False,device='cuda').load(load_path + '/best_model.zip',device='cuda') 
        partner = StaticPolicyAgent(previous_policy.policy)
        env.add_partner_agent(partner, player_num=1)

        order = expert_examples['start'][0]
        env.base_env.set_reset_point(order)
        time.sleep(1)
        env.reset()
        actions=[i for i in expert_examples['actions']]
        print(actions)


        visualShapeId = pybullet_instance.createVisualShape(shapeType=pybullet_instance.GEOM_CYLINDER,
                                            rgbaColor=[1, 0, 0, 1],
                                            radius=0.004,
                                            length=0.02,
                                            specularColor=[0.4, .4, 0],
                                            visualFramePosition=[[expert_examples['goals'][0][0][0],expert_examples['goals'][0][0][1]+0.1,0.1]],
                                            visualFrameOrientation=[ 0.7071068, 0, 0, 0.7071068 ])
        collisionShapeId = pybullet_instance.createCollisionShape(shapeType=pybullet_instance.GEOM_CYLINDER,
                                                radius=0.002,
                                                height=0.002,)

        tting = pybullet_instance.createMultiBody(baseMass=0,
                        baseInertialFramePosition=[0,0,0],
                        baseCollisionShapeIndex=collisionShapeId,
                        baseVisualShapeIndex=visualShapeId,
                        basePosition=[expert_examples['goals'][0][0][0]-0.0025,expert_examples['goals'][0][0][1]+0.1-0.0025,0.15],
                        baseOrientation =[0,0,0,1],
                        useMaximalCoordinates=True)

        visualShapeId1 = pybullet_instance.createVisualShape(shapeType=pybullet_instance.GEOM_CYLINDER,
                                            rgbaColor=[1, 0, 0, 1],
                                            radius=0.004,
                                            length=0.02,
                                            specularColor=[0.4, .4, 0],
                                            visualFramePosition=[[expert_examples['goals'][0][5][0],expert_examples['goals'][0][5][1]+0.1,0.1]],
                                            visualFrameOrientation=[ 0.7071068, 0, 0, 0.7071068 ])
        collisionShapeId1 = pybullet_instance.createCollisionShape(shapeType=pybullet_instance.GEOM_CYLINDER,
                                                radius=0.002,
                                                height=0.002,)

        tting1 = pybullet_instance.createMultiBody(baseMass=0,
                        baseInertialFramePosition=[0,0,0],
                        baseCollisionShapeIndex=collisionShapeId1,
                        baseVisualShapeIndex=visualShapeId1,
                        basePosition=[expert_examples['goals'][0][5][0]-0.0025,expert_examples['goals'][0][5][1]+0.1-0.0025,0.15],
                        baseOrientation =[0,0,0,1],
                        useMaximalCoordinates=True)
        
        visualShapeId2 = pybullet_instance.createVisualShape(shapeType=pybullet_instance.GEOM_CYLINDER,
                                            rgbaColor=[1, 0, 0, 1],
                                            radius=0.004,
                                            length=0.02,
                                            specularColor=[0.4, .4, 0],
                                            visualFramePosition=[[expert_examples['goals'][0][10][0],expert_examples['goals'][0][10][1]+0.1,0.1]],
                                            visualFrameOrientation=[ 0.7071068, 0, 0, 0.7071068 ])
        collisionShapeId2 = pybullet_instance.createCollisionShape(shapeType=pybullet_instance.GEOM_CYLINDER,
                                                radius=0.002,
                                                height=0.002,)

        tting2 = pybullet_instance.createMultiBody(baseMass=0,
                        baseInertialFramePosition=[0,0,0],
                        baseCollisionShapeIndex=collisionShapeId2,
                        baseVisualShapeIndex=visualShapeId2,
                        basePosition=[expert_examples['goals'][0][10][0]-0.0025,expert_examples['goals'][0][10][1]+0.1-0.0025,0.15],
                        baseOrientation =[0,0,0,1],
                        useMaximalCoordinates=True)
        
        visualShapeId3 = pybullet_instance.createVisualShape(shapeType=pybullet_instance.GEOM_CYLINDER,
                                            rgbaColor=[1, 0, 0, 1],
                                            radius=0.004,
                                            length=0.02,
                                            specularColor=[0.4, .4, 0],
                                            visualFramePosition=[[expert_examples['goals'][0][15][0],expert_examples['goals'][0][15][1]+0.1,0.1]],
                                            visualFrameOrientation=[ 0.7071068, 0, 0, 0.7071068 ])
        collisionShapeId3 = pybullet_instance.createCollisionShape(shapeType=pybullet_instance.GEOM_CYLINDER,
                                                radius=0.002,
                                                height=0.002,)

        tting3 = pybullet_instance.createMultiBody(baseMass=0,
                        baseInertialFramePosition=[0,0,0],
                        baseCollisionShapeIndex=collisionShapeId3,
                        baseVisualShapeIndex=visualShapeId3,
                        basePosition=[expert_examples['goals'][0][15][0]-0.0025,expert_examples['goals'][0][15][1]+0.1-0.0025,0.15],
                        baseOrientation =[0,0,0,1],
                        useMaximalCoordinates=True)
        
        visualShapeId4 = pybullet_instance.createVisualShape(shapeType=pybullet_instance.GEOM_CYLINDER,
                                            rgbaColor=[1, 0, 0, 1],
                                            radius=0.004,
                                            length=0.02,
                                            specularColor=[0.4, .4, 0],
                                            visualFramePosition=[[expert_examples['goals'][0][20][0],expert_examples['goals'][0][20][1]+0.1,0.1]],
                                            visualFrameOrientation=[ 0.7071068, 0, 0, 0.7071068 ])
        collisionShapeId4 = pybullet_instance.createCollisionShape(shapeType=pybullet_instance.GEOM_CYLINDER,
                                                radius=0.002,
                                                height=0.002,)

        tting4 = pybullet_instance.createMultiBody(baseMass=0,
                        baseInertialFramePosition=[0,0,0],
                        baseCollisionShapeIndex=collisionShapeId4,
                        baseVisualShapeIndex=visualShapeId4,
                        basePosition=[expert_examples['goals'][0][20][0]-0.0025,expert_examples['goals'][0][20][1]+0.1-0.0025,0.15],
                        baseOrientation =[0,0,0,1],
                        useMaximalCoordinates=True)
        time.sleep(1)
        for i in range(25):
            env.step(np.array(actions[i]))
            goals = expert_examples['goals'][0][i]
            print('goals',goals)
            time.sleep(0.05)

if __name__ == '__main__':
    # train_expert_multiprocess('./data/HRL_image_pretrained_frozen/experiment_config.json',num_cpu=16)
    # train_expert('./data/HRL_image_pretrained_expert/experiment_config.json')
    # test_expert_data('./data/HRL_image_pretrained_expert/experiment_config.json')
    # train_expert('./data/HRL_froze_pretrained_worker_expert/experiment_config.json')
    # [[0.03,0.024],[-0.05,0.0367],[-0.06,-0.023],[0.02,-0.05],[0.05,0.02],[-0.003,0.05]]
    # train_expert('./data/HRL_simplified_pretrained_expert/experiment_config.json')
    train_expert('./data/HRL_simplified_expert_no_pretrained/experiment_config.json')
    # train_expert('./data/HRL_simplified_none/experiment_config.json')
    # train_expert('./data/HRL_simplified_no_expert/experiment_config.json')
    
    # TODOs
    # try the thing with pretraining and all that
    # try without the fucking stuff