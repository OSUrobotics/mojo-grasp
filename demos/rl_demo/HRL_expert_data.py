import sys
sys.path.append('/home/mothra/pybullet-planning')
# import pybullet_tools.utils as pp

from demos.rl_demo.learnpettingzooHRL import *
from scipy.interpolate import interp2d
import torch 
from stable_baselines3.common.buffers import ReplayBuffer, RolloutBuffer
from demos.rl_demo.policies import PPOExpertData


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
    for i in range(int(len(poses[0])/100)-4):
        goal_poses = poses[0][i:i+5].tolist()
        start_point = poses[0][np.random.randint(len(poses[0]))]
        order = [start_point.tolist()]
        for _ in range(5):
            dists = np.linalg.norm(start_point - np.array(goal_poses),axis=1)
            # print(dists, start_point, np.array(goal_poses))
            next_ind = np.argmin(dists)
            order.append(goal_poses[next_ind])
            start_point = np.array(goal_poses[next_ind])
            goal_poses.pop(next_ind)
        pose_orders.append(order)

    buffer = RolloutBuffer(25*len(pose_orders),env.observation_space('manager'),env.action_space('manager'))
    just_goals = {'goals':[],'actions':[], 'start':[]}
    for order in pose_orders:
        # go through each of the orders we made earlier and set the simulator to those points
        # using 5 steps per spot as the goal point, starting the object at the desired point
        # and interpolating from that spot
        env.reset({'start_pos':order[0],'finger_angs':[-np.pi/2,0,np.pi/2,0]})
        goals=[order[int(i/5)+1] for i in range(25)]
        order_array = np.array(order)
        object_poses = [np.linspace(order_array[i],order_array[i+1],5) for i in range(len(order_array)-1)]
        object_poses = np.array(object_poses)
        object_poses = object_poses.reshape(25,2)

        for i in range(25):
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
    env.train()
    return buffer, just_goals


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
    test_expert_data('./data/HRL_image_pretrained_expert/experiment_config.json')
    # train_expert('./data/HRL_froze_pretrained_worker_expert/experiment_config.json')
    # [[0.03,0.024],[-0.05,0.0367],[-0.06,-0.023],[0.02,-0.05],[0.05,0.02],[-0.003,0.05]]