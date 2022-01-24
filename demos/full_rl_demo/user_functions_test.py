import mojograsp
from tensorboardX import SummaryWriter


class UserFunctionsTest(mojograsp.user_functions_base.UserFunctionsBase):
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
        self.training_phase = data['move rl']
        self.training_phase.controller.load('saved_weights')

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
