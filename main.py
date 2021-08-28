import gym
import gym_env_files
import mojograsp.simcore.simmanager.state_space

# env = gym.make('ihm-v1')
a = mojograsp.simcore.simmanager.state_space.StateSpace()
print(a.get_full_arr())