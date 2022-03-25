'''
FedAvg算法
'''
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn


def FedAvgV1(w, total_data_sum=0, user_idx_this_round=[1, 2, 3, 4], dict_users=None):
    '''
    考虑了datasize权重的FedAvg
    :param w:
    :param total_data_sum:
    :param user_idx_this_round:
    :param dict_users:
    :return:
    '''
    w_avg = copy.deepcopy(w[0])     # 把第'0'个local_weight拿出来
    print("w_avg: ", type(w_avg))   # <class 'collections.OrderedDict'>
    for k in w_avg.keys():
        w_avg[k] = torch.mul(w_avg[k], len(dict_users[user_idx_this_round[0]]))     # local_weight乘以根据数据量算出来的权重
        j = 1
        for i in range(1, len(user_idx_this_round)):
            datasize = len(dict_users[user_idx_this_round[i]])
            w_avg[k] += torch.mul(w[j][k], datasize)    # * len(dict_users[user_idx_this_round[i]])
            j += 1
        w_avg[k] = torch.div(w_avg[k], total_data_sum)
    return w_avg


def FedAvg(w):
    len_of_w = len(w)
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len_of_w):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len_of_w)
    return w_avg
