#!/usr/bin/env python3


class EnvironmentBase:
    """
    This class is the base class to write methods to reset the simulator's environment, execute actions and step the
    simulator ahead.
    To interact with more items in the simulator, such as getting object position, joint angles, etc add methods here.

    Note: For an environment class that uses gym, derived from this base class, you need to inherit from gym as well.
    Example: class EnvironmentMujoco(EnvironmentBase, gym.Env)
    """

    def __init__(self, action_class=None):
        """
        Initializes an instance of this class. This class need values of simulation step and rl step, hence would
        require interaction with the action_class.
        :param action_class: An instance of action class, to get simulation and rl timestep informaiton.
        """
        self.rl_step = None
        self.sim_step = None
        pass

    def reset(self):
        """
        :return:
        """
        pass

    def step(self, phase):
        """
        Interacts with the action class to build the action profile,
        calls on step_sim to step the simulator depending on the rl steps and simulator steps.
        :param phase:
        :return:
        """
        observation = None
        reward = None
        done = None
        info = None
        #call on action class to build action profile

        #pass action profile to step_sim to take action
        self.step_sim(phase.action.action_profile)
        return observation, reward, done, info

    def step_sim(self, action_profile):
        """
        Take an action based on the profile every simulator time step to build the action.
        :param action_profile:
        :return:
        """
        pass
