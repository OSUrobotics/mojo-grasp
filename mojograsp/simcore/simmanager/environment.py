#!/usr/bin/env python3
import time
import pybullet as p


class Environment:
    """
    This class is used to reset the simulator's environment, execute actions, and step the simulator ahead
    """

    def __init__(self, sleep=1. / 240., steps=1):
        self.sleep = sleep
        self.steps = steps
        pass

    def reset(self):
        pass

    def step(self, vals):
        phase = vals[0]
        state = vals[1]
        reward = vals[2]
        phase.execute_action(phase.action)
        self.step_sim()
        print("State:", state)
        if state is not None:
            state.update()
            observation = state.get_observation()
        else:
            observation = None
        return observation, None, None, None

    def step_sim(self):
        for i in range(self.steps):
            p.stepSimulation()
            time.sleep(self.sleep)
