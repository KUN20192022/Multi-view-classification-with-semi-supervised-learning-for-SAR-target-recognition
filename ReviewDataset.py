# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 10:36:09 2020

@author: Administrator
"""

class ReviewDataset():
    def __init__(self, data_list, target_list, weights_list, class_weights, max_sent_length):
        self.data_list = data_list
        self.target_list = target_list
        self.max_sent_length = max_sent_length
        self.weights_list = weights_list
        self.class_weights = class_weights
        assert (len(self.data_list) == len(self.target_list))
        assert (len(self.weights_list) == len(self.data_list))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, key, max_sent_length=None):
        if max_sent_length is None:
            max_sent_length = self.max_sent_length

        token_idx = self.data_list[key][:max_sent_length]#创建迭代器
        label = self.target_list[key]
        w = self.weights_list[key]
        c = self.class_weights[label]

        return [token_idx, label, w, c]