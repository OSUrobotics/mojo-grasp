from mojograsp.simcore.action import Action


class ExpertAction(Action):
    def __init__(self):
        self.current_action = {}

    def set_action(self, joint_angles: list, actor_output: list):
        self.current_action["target_joint_angles"] = joint_angles
        self.current_action['actor_output'] = actor_output

    def get_action(self) -> dict:
        return self.current_action.copy()

class IKAction(Action):
    def __init__(self):
        self.current_action = {}

    def set_action(self, joint_angles: list):
        self.current_action["target_joint_angles"] = joint_angles
        

    def get_action(self) -> dict:
        return self.current_action.copy()