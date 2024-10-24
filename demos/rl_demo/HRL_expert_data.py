import sys
sys.path.append('/home/mothra/pybullet-planning')
# import pybullet_tools.utils as pp

from demos.rl_demo.learnpettingzooHRL import *
from scipy.interpolate import interp2d
import torch 
from stable_baselines3.common.buffers import ReplayBuffer, RolloutBuffer
from demos.rl_demo.policies import PPOExpertData

def collect_expert(env,poses):
    env.evaluate()
    # start small
    
    pose_orders = []
    for i in range(int(len(poses[0]))-4):
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
            # print(goals[i], action, )
            assert all(abs(action)< 1)  
            
            # normalized_action = (action+1) * self.manager_normalizer['diff']/2 + self.manager_normalizer['mins']
            reward = env.rewards['manager']
            log_probs = torch.zeros(1)
            values=torch.zeros(1)
            buffer.add(observation,action,reward,i==0,values,log_probs)
    env.train()
    return buffer


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
        expert_buffer = collect_expert(env,poses)
    env = PettingZooAECWrapper(env)
    if args['load_path'] == '//':
        partner = OnPolicyAgent(PPO('MlpPolicy', env.getDummyEnv(1), verbose=1, tensorboard_log=args['tname']+'/worker'), tensorboard_log=args['tname']+'/worker')
    else:
        load_path = args['load_path']
        previous_policy = model_type("MlpPolicy", None, _init_setup_model=False,device='cpu').load(load_path + '/best_model.zip',device='cpu') 
        partner = StaticPolicyAgent(previous_policy)

    # The second parameter ensures that the partner is assigned to a certain
    # player number. Forgetting this parameter would mean that all of the
    # partner agents can be picked as `player 2`, but none of them can be
    # picked as `player 3`. 
    env.add_partner_agent(partner, player_num=1)
    train_timesteps = int(args['evaluate']*(args['tsteps']+1))
    worker_callback = pettingzoowrapper.WorkerEvaluateCallback(partner.model, args['save_path'])
    callback = pettingzoowrapper.ZooEvaluateCallback(env,n_eval_episodes=int(1200), eval_freq=int(train_timesteps), best_model_save_path=args['save_path'],callback_on_new_best=worker_callback)

    if args['model']== "PPO":
        model = model_type("MlpPolicy", env,tensorboard_log=args['tname']+'/manager')
    elif args['model'] == "PPO_Expert":
        model = model_type("MlpPolicy", env,tensorboard_log=args['tname']+'/manager',expert_buffer=expert_buffer)
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

if __name__ == '__main__':
    train_expert('./data/HRL_multigoal_expert_x100/experiment_config.json')
    # collect_expert('./data/HRL_multigoal_fixed/experiment_config.json')