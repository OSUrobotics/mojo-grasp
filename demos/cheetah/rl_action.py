from mojograsp.simcore.action import Action


class ExpertAction(Action):
    def __init__(self):
        self.current_action = {}

    def set_action(self, joint_angles: list):
        self.current_action["target_joint_angles"] = joint_angles

    def get_action(self) -> dict:
        return self.current_action.copy()
