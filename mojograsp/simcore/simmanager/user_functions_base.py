class UserFunctionsBase:
    _sim = None

    def __init__(self, data=None):
        self.data = data
        print ("User functions")

    def pre_run(self, data=None):
        pass

    def post_run(self, data=None):
        pass

    def pre_phaseloop(self, data=None):
        pass

    def post_phaseloop(self, data=None):
        pass

    def pre_phase(self, data=None):
        pass

    def post_phase(self, data=None):
        pass

    def post_lastphase(self, data=None):
        pass
