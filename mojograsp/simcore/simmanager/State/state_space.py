#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 14:57:24 2021
@author: orochi
"""

from mojograsp.simcore.simmanager.State.state_space_base import StateSpaceBase


class StateSpace(StateSpaceBase):

    def __init__(self, path=None):
        super().__init__(path)
        for name, value in self.json_data.items():
            state_name = name.split(sep='_')
            try:
                # print('State Name: {}, Value: {}, Name: {}'.format(state_name[0], value, name))
                self.data[name] = StateSpace.valid_state_names[state_name[0]](value)
            except NameError:
                print(state_name[0], 'Invalid state name. Valid state names are', [name for name in
                                                                                   StateSpace.valid_state_names.keys()])

    def get_obs(self):
        arr = []
        for name, value in self.data.items():
            temp = value.get_value()
            try:
                arr.extend(temp)
            except TypeError:
                arr.extend([temp])
        return arr

    def get_value(self, keys):
        if type(keys) is str:
            keys = [keys]
        # print("KEYS: {} {} \n{}".format(keys, keys[0], self.data))
        if len(keys) > 1:
            data = self.data[keys[0]].get_specific(keys[1:])
        else:
            data = self.data[keys[0]].get_value()
        return data

    def update(self):
        for name, value in self.data.items():
            # print("KEY IS: {}, {}".format(name, self.data[name]))
            self.data[name].update(name)
        return self.get_obs()


if __name__ == "__main__":
    try_state = StateSpace()

    try_state.update()
    pass
