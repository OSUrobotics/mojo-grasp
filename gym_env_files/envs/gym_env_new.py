import gym
from mojograsp.simcore.simmanager import environment
from mojograsp.simcore.simmanager.Reward import reward_class


class IHMBulletEnv(gym.Env):

    def __init__(self):
        self.sim = environment.Environment()
        # self.state_space = state
        # self.reward_class = reward
        self.done = False
        self.info = None

        pass

    def reset(self):
        """
        Should  reset  object  pose  and   robot fingers  back to starting position
        :return:
        """
        pass

    def step(self, vals):#phase, state, reward):
        """
        Takes a step in pybullet env
        :param action: should be joint angle changes as a list type
        :return:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)

        """
        phase = vals[0]
        state = vals[1]
        reward_here = vals[2]
        print("REWARD POINTER:", reward_here)

        observation, _, _, _ = self.sim.step(vals)

        # Get reward
        print("OBSERVATION:", observation)
        # curr_reward = reward_here.get_reward()
        curr_reward = 0
        print("Observation: {}\nReward: {}\nDone: {}\nInfo:{}\nMax Episodes:".format(observation, curr_reward, self.done, self.info))#, self.max_episode_steps))
        return observation, curr_reward, self.done, self.info
        pass

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        pass


if __name__ == "__main__":
    pass
