from abc import ABC, abstractmethod
import json
import logging
from mojograsp.simcore.action import Action, ActionDefault
from mojograsp.simcore.reward import Reward, RewardDefault
from mojograsp.simcore.state import State, StateDefault
from mojograsp.simcore.replay_buffer import ReplayBuffer
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import pickle as pkl

import operator
import time

# https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py using openai implementation of the segment/sum tree


class SegmentTree(object):
    def __init__(self, capacity, operation, neutral_element):
        """Build a Segment Tree data structure.
        https://en.wikipedia.org/wiki/Segment_tree
        Can be used as regular array, but with two
        important differences:
            a) setting item's value is slightly slower.
               It is O(lg capacity) instead of O(1).
            b) user has access to an efficient ( O(log segment size) )
               `reduce` operation which reduces `operation` over
               a contiguous subsequence of items in the array.
        Paramters
        ---------
        capacity: int
            Total size of the array - must be a power of two.
        operation: lambda obj, obj -> obj
            and operation for combining elements (eg. sum, max)
            must form a mathematical group together with the set of
            possible values for array elements (i.e. be associative)
        neutral_element: obj
            neutral element for the operation above. eg. float('-inf')
            for max and 0 for sum.
        """
        assert capacity > 0 and capacity & (
            capacity - 1) == 0, "capacity must be positive and a power of 2."
        self._capacity = capacity
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self._operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(
                        mid + 1, end, 2 * node + 1, mid + 1, node_end)
                )

    def reduce(self, start=0, end=None):
        """Returns result of applying `self.operation`
        to a contiguous subsequence of the array.
            self.operation(arr[start], operation(arr[start+1], operation(... arr[end])))
        Parameters
        ----------
        start: int
            beginning of the subsequence
        end: int
            end of the subsequences
        Returns
        -------
        reduced: obj
            result of reducing self.operation over the specified range of array elements.
        """
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        # index of the leaf
        idx += self._capacity
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(
                self._value[2 * idx],
                self._value[2 * idx + 1]
            )
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(
            capacity=capacity,
            operation=operator.add,
            neutral_element=0.0
        )

    def sum(self, start=0, end=None):
        """Returns arr[start] + ... + arr[end]"""
        return super(SumSegmentTree, self).reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        """Find the highest index `i` in the array such that
            sum(arr[0] + arr[1] + ... + arr[i - i]) <= prefixsum
        if array values are probabilities, this function
        allows to sample indexes according to the discrete
        probability efficiently.
        Parameters
        ----------
        perfixsum: float
            upperbound on the sum of array prefix
        Returns
        -------
        idx: int
            highest index satisfying the prefixsum constraint
        """
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        while idx < self._capacity:  # while non-leaf
            if self._value[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._capacity


class ReplayBufferPriority():
    def __init__(self, buffer_size: int = 39999, alpha: float = 0.3, beta: float = 1.0, max_prio: float = 1):
        self.buffer_memory = np.zeros(buffer_size, dtype=object)
        pwr_sz = 1
        while pwr_sz < buffer_size:
            pwr_sz *= 2
        self.buffer_prio = SumSegmentTree(pwr_sz)

        self.idx = 0
        self.sz = 0
        self.demo_sz = 0
        self.max_prio = max_prio
        self.alpha = alpha
        self.beta = beta

        self.buffer_max = buffer_size

        self.rand = np.random.RandomState(None)
        self.sampled_indexes = []

    def preload_buffer_PKL(self, file_name):
        with open(file_name, 'rb') as handle:
            b = pkl.load(handle)

        prev = None
        for i in b["episode_list"]:
            # print('replay buffer size', self.sz)
            for j in i["timestep_list"]:
                if prev:
                    # print(i['number'])
                    if prev[-2] == i["number"]:
                        # print(prev)
                        temp = list(prev)
                        temp[-3] = j["state"]
                        prev = tuple(temp)
                        # print(prev[3])
                        if self.sz < self.buffer_max:
                            self.buffer_memory[self.idx] = prev
                            self.buffer_prio[self.idx] = self.max_prio
                            self.idx += 1
                            self.demo_sz += 1
                            self.sz += 1
                        else:
                            print(
                                "ERROR: REPLAY BUFFER OUT OF SPACE, TRANSITION NOT ADDED")
                prev = (j["state"], j["action"],
                        j["reward"], None, i["number"], 1)
                
        # if self.sz < self.buffer_max:
        #     self.buffer_memory[self.idx] = prev
        #     self.buffer_prio[self.idx] = self.max_prio
        #     self.idx += 1
        #     self.demo_sz += 1
        #     self.sz += 1
        # else:
        #     print("ERROR: REPLAY BUFFER OUT OF SPACE, TRANSITION NOT ADDED")
        pass

    def sample_rollout(self, batch_size: int, rollout_size: int):
        b_size = min(self.sz-1, batch_size)
        prio_sum = self.buffer_prio.sum(0, self.sz)

        idxes = []
        transitions = []
        weights = []
        rnums = self.rand.uniform(0, prio_sum, b_size)
        for i in range(b_size):
            r_num = rnums[i]
            sample_idx = self.buffer_prio.find_prefixsum_idx(r_num)
            
            if sample_idx not in idxes:
                temp_idx = []
                temp_trans = []
                temp_w = []
                # self.sampled_indexes.append(sample_idx)
                for i in range(rollout_size):
                    if sample_idx + i <= self.sz-1:
                        rollout_sample = self.buffer_memory[sample_idx + i]
                        if rollout_sample != 0 and rollout_sample[4] == self.buffer_memory[sample_idx][4]:
                            temp_idx.append(sample_idx + i)
                            temp_trans.append(self.buffer_memory[sample_idx + i])
                            wt = (
                                (1/(self.buffer_prio[sample_idx + i] / prio_sum)) * (1/self.sz)) ** self.beta
                            # print(type(wt),wt)
                            temp_w.append(wt)
                # print('temp w type', type(temp_w), type(temp_w[0]))
                idxes.append(temp_idx)
                transitions.append(temp_trans)
                weights.append(temp_w)
        return transitions, weights, idxes

    def sample_sequence_rollout(self, batch_size: int, prestep_size: int, rollout_size: int):
        b_size = min(self.sz-1, batch_size)
        prio_sum = self.buffer_prio.sum(0, self.sz)

        idxes = []
        transitions = []
        weights = []
        lookback = []
        
        rnums = self.rand.uniform(0, prio_sum, b_size)
        for i in range(b_size):
            r_num = rnums[i]
            sample_idx = self.buffer_prio.find_prefixsum_idx(r_num)
            
            if sample_idx not in idxes:
                temp_idx = []
                temp_trans = []
                temp_w = []
                temp_prevs = []
                episode_num = self.buffer_memory[sample_idx][4]
                earliest_transition = self.buffer_memory[sample_idx]
                for i in range(1,prestep_size+1):
                    rollout_sample = self.buffer_memory[sample_idx - i]
                    if rollout_sample != 0 and rollout_sample[4] == episode_num:
                        temp_prevs.append(self.buffer_memory[sample_idx - i])
                        earliest_transition = self.buffer_memory[sample_idx - i]
                    else:
                        temp_prevs.append(earliest_transition)
                for i in range(rollout_size):
                    if sample_idx + i <= self.sz-1:
                        rollout_sample = self.buffer_memory[sample_idx + i]
                        if rollout_sample != 0 and rollout_sample[4] == self.buffer_memory[sample_idx][4]:
                            temp_idx.append(sample_idx + i)
                            temp_trans.append(self.buffer_memory[sample_idx + i])
                            wt = (
                                (1/(self.buffer_prio[sample_idx + i] / prio_sum)) * (1/self.sz)) ** self.beta
                            temp_w.append(wt)
                idxes.append(temp_idx)
                transitions.append(temp_trans)
                
                weights.append(temp_w)
                lookback.append(temp_prevs)
        return transitions, weights, idxes, lookback


    def update_priorities(self, idxes, priorities):
        for i in range(len(idxes)):
            self.buffer_prio[idxes[i]] = float(priorities[i] ** self.alpha)
            
            # self.buffer_prio[idxes[i]] = float(self.buffer_prio[idxes[i]])
            self.max_prio = float(max(priorities[i], self.max_prio))
            

    def add_timestep(self, transition):
        if self.sz < self.buffer_max:
            self.buffer_memory[self.idx] = transition
            self.buffer_prio[self.idx] = self.max_prio
            self.idx += 1
            self.sz += 1
        else:
            print("ERROR: REPLAY BUFFER OUT OF SPACE, TRANSITION NOT ADDED")

    def save_buffer(self, filename: str):
        pass

    def save_sampling(self, filename: str):
        with open(filename+'.pkl', 'wb') as file:
            pkl.dump(self.sampled_indexes, file)

    def __len__(self):
        return self.sz