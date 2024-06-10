#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 13:16:39 2024

@author: orochi
"""
import numpy as np
from mojograsp.simcore.state import State

class modelHolder:
    def __init__(self, model,args):
        self.model = model
        self.state_list = args['state_list']
        self.PREV_VALS = args['pv']

    def build_state(self, state_container: State):
        """
        Method takes in a State object 
        Extracts state information from state_container and returns it as a list based on
        current used states contained in self.state_list

        :param state: :func:`~mojograsp.simcore.phase.State` object.
        :type state: :func:`~mojograsp.simcore.phase.State`
        """
        angle_keys = ["finger0_segment0_joint","finger0_segment1_joint","finger1_segment0_joint","finger1_segment1_joint"]
        state = []
        if self.PREV_VALS > 0:
            for i in range(self.PREV_VALS):
                for key in self.state_list:
                    if key == 'op':
                        state.extend(state_container['previous_state'][i]['obj_2']['pose'][0][0:2])
                    elif key == 'oo':
                        state.extend(state_container['previous_state'][i]['obj_2']['pose'][1])
                    elif key == 'oa':
                        state.extend([np.sin(state_container['previous_state'][i]['obj_2']['z_angle']),np.cos(state_container['previous_state'][i]['obj_2']['z_angle'])])
                    elif key == 'ftp':
                        state.extend(state_container['previous_state'][i]['f1_pos'][0:2])
                        state.extend(state_container['previous_state'][i]['f2_pos'][0:2])
                    elif key == 'fbp':
                        state.extend(state_container['previous_state'][i]['f1_base'][0:2])
                        state.extend(state_container['previous_state'][i]['f2_base'][0:2])
                    elif key == 'fcp':
                        state.extend(state_container['previous_state'][i]['f1_contact_pos'][0:2])
                        state.extend(state_container['previous_state'][i]['f2_contact_pos'][0:2])
                    elif key == 'ja':
                        state.extend([state_container['previous_state'][i]['two_finger_gripper']['joint_angles'][item] for item in angle_keys])
                    elif key == 'fta':
                        state.extend([state_container['previous_state'][i]['f1_ang'],state_container['previous_state'][i]['f2_ang']])
                    elif key == 'eva':
                        state.extend(state_container['previous_state'][i]['two_finger_gripper']['eigenvalues'])
                    elif key == 'evc':
                        state.extend(state_container['previous_state'][i]['two_finger_gripper']['eigenvectors'])
                    elif key == 'evv':
                        evecs = state_container['previous_state'][i]['two_finger_gripper']['eigenvectors']
                        evals = state_container['previous_state'][i]['two_finger_gripper']['eigenvalues']
                        scaled = [evals[0]*evecs[0],evals[0]*evecs[2],evals[1]*evecs[1],evals[1]*evecs[3],
                                  evals[2]*evecs[4],evals[2]*evecs[6],evals[3]*evecs[5],evals[3]*evecs[7]]
                        state.extend(scaled)
                    elif key == 'params':
                        state.extend(state_container['hand_params'])
                    elif key == 'gp':
                        state.extend(state_container['previous_state'][i]['goal_pose']['goal_position'])
                    elif key == 'go':
                        state.append(state_container['previous_state'][i]['goal_pose']['goal_orientation'])
                    elif key == 'gf':
                        state.extend(state_container['previous_state'][i]['goal_pose']['goal_finger'])
                    else:
                        raise Exception('key does not match list of known keys')

        for key in self.state_list:
            if key == 'op':
                state.extend(state_container['obj_2']['pose'][0][0:2])
            elif key == 'oo':
                state.extend(state_container['obj_2']['pose'][1])
            elif key == 'oa':
                state.extend([np.sin(state_container['obj_2']['z_angle']),np.cos(state_container['obj_2']['z_angle'])])
            elif key == 'ftp':
                state.extend(state_container['f1_pos'][0:2])
                state.extend(state_container['f2_pos'][0:2])
            elif key == 'fbp':
                state.extend(state_container['f1_base'][0:2])
                state.extend(state_container['f2_base'][0:2])
            elif key == 'fcp':
                state.extend(state_container['f1_contact_pos'][0:2])
                state.extend(state_container['f2_contact_pos'][0:2])
            elif key == 'ja':
                state.extend([state_container['two_finger_gripper']['joint_angles'][item] for item in angle_keys])
            elif key == 'fta':
                state.extend([state_container['f1_ang'],state_container['f2_ang']])
            elif key == 'eva':
                state.extend(state_container['two_finger_gripper']['eigenvalues'])
            elif key == 'evc':
                state.extend(state_container['two_finger_gripper']['eigenvectors'])
            elif key == 'evv':
                evecs = state_container['two_finger_gripper']['eigenvectors']
                evals = state_container['two_finger_gripper']['eigenvalues']
                scaled = [evals[0]*evecs[0],evals[0]*evecs[2],evals[1]*evecs[1],evals[1]*evecs[3],
                          evals[2]*evecs[4],evals[2]*evecs[6],evals[3]*evecs[5],evals[3]*evecs[7]]
                state.extend(scaled)
            elif key == 'params':
                state.extend(state_container['hand_params'])
            elif key == 'gp':
                state.extend(state_container['goal_pose']['goal_position'])
            elif key == 'go':
                state.append(state_container['goal_pose']['goal_orientation'])
            elif key == 'gf':
                state.extend(state_container['goal_pose']['goal_finger'])
            else:
                raise Exception('key does not match list of known keys')
        return state

    def __call__(self, State_vec):
        prepped = self.build_state(State_vec)
        return self.model.predict(prepped)