import mojograsp
from tensorboardX import SummaryWriter


class UserFunctionsTrain(mojograsp.user_functions_base.UserFunctionsBase):
    def __init__(self, data=None):
        """

        """
        super().__init__(data)
        self.writer = None
        self.training_phase = None
        self.replay_expert = None
        self.replay_agent = None
        self.actor_loss, self.critic_loss, self.critic_L1loss, self.critic_LNloss = None, None, None, None

    def pre_run(self, data=None):
        """

        :param data: Pass in the phase manager's phase dictionary object
        :return:
        """
        self.writer = SummaryWriter(log_dir='/Users/asar/Desktop/Grimm\'s Lab/Manipulation/PyBulletStuff/mojo-grasp/demos/full_rl_demo/data')
        self.training_phase = data['move rl']

    def post_lastphase(self, data=None):
        """
        :param data: data is a list of parameters required for this method.
                0th index is the episode number
                1st index is the replay buffer of expert controller
                2nd index is the replay buffer of agent controller

        :return: pass
        """
        i = data[0]
        self.replay_expert = data[1]
        self.replay_agent = data[2]

        # Training of network (Everything inside if statement. Comment while evaluating)
        if i > 20:
            self.actor_loss, self.critic_loss, self.critic_L1loss, self.critic_LNloss = self.training_phase.controller.train_batch(
                                                                    max_episode_num=self.training_phase.terminal_step,
                                                                    episode_num=i, update_count=1,
                                                                    expert_replay_buffer=self.replay_expert,
                                                                    replay_buffer=self.replay_agent)

    def post_phaseloop(self, data=None):
        """
        data is a list that contains episode number and reward values
        :param 0th index is the episode number
               1st index is the reward
        :return:
        """
        i = data[0]
        reward = data[1]
        print(i)

        if not (i % 200) and i > 20:
            # if not (i % 1):
            self.writer.add_scalar("Final Reward", reward[0], i)
            self.writer.add_scalar("Actor Loss", self.actor_loss, i)
            self.writer.add_scalar("Critic Loss", self.critic_loss, i)
            self.writer.add_scalar("Critic_L1 Loss", self.critic_L1loss, i)
            self.writer.add_scalar("Critic_LN Loss", self.critic_LNloss, i)

        if not (i % 2000):
            self.training_phase.controller.save('saved_weights')

    def post_run(self, data=None):
        print("Saving weights...")
        self.training_phase.controller.save('saved_weights')
