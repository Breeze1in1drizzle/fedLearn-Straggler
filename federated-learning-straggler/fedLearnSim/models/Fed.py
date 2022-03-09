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
    len_of_w = len(w)       # w的长度，即local_weights的数量
    # print("user_idx_this_round: ", len(user_idx_this_round))
    # print("len_of_w: ", len_of_w)
    w_avg = copy.deepcopy(w[0])     # 把第'0'个local_weight拿出来
    # ONE_DIV_datasize_weight = float(total_data_sum/len(dict_users[user_idx_this_round[0]]))    # Float类型 第0个local_weight对应的FedAvg的加和权重（自己的datasize/所有datasize之和）
    # # print("datasize_weight: ", type(datasize_weight), ", ", datasize_weight)
    # print("w_avg: ", type(w_avg))       # w_avg:  <class 'collections.OrderedDict'>
    # w_avg = torch.div(w_avg,
    #                   1.0/float(len(dict_users[user_idx_this_round[0]])))     # local_weight乘以根据数据量算出来的权重
    # print("w_avg.keys():\n", w_avg.keys())
    # count_k = 0
    for k in w_avg.keys():
        # for i in range(1, len_of_w):  # w[i]就是一个local_weights eg.len_of_w==4-->4个有效participants
        #     w_avg[k] += w[i][k]     # 需要优化——考虑datasize的比例
        # print("w_avg.keys():\n", w_avg.keys())
        # if count_k == 0:
        # print("datasize: ", len(dict_users[user_idx_this_round[0]]))
        w_avg[k] = torch.mul(w_avg[k], len(dict_users[user_idx_this_round[0]]))     # local_weight乘以根据数据量算出来的权重

        j = 1
        for i in range(1, len(user_idx_this_round)):
            # print("w[", j, "][", k, "] type: ", type(w[j][k]))
            datasize = len(dict_users[user_idx_this_round[i]])
            # print("datasize type: ", type(datasize))
            # print("datasize: ", datasize, ", i: ", i)
            # 这里的不同的k是遍历同一个模型的不同layer：conv1,conv2......
            # datasize_weight = ONE_DIV_datasize_weight = float(total_data_sum/len(dict_users[user_idx_this_round[0]]))
            w_avg[k] += torch.mul(w[j][k], datasize)# * len(dict_users[user_idx_this_round[i]])
            j += 1
        # print("w_avg[k]: ", type(w_avg[k]))
        # print("total_data_sum: ", total_data_sum)
        w_avg[k] = torch.div(w_avg[k], total_data_sum)
        # count_k += 1
    return w_avg


def FedAvgTest(w):
    len_of_w = len(w)
    w_avg = copy.deepcopy(w[0])
    # print("w_avg: ", type(w_avg))     # w_avg:  <class 'collections.OrderedDict'>
    for k in w_avg.keys():
        # print("k: ", k, ", ", type(k))
        # print("before for loop w_avg[k]: ", type(w_avg[k]))
        # w_avg[k] = torch.mul(w_avg[k], int(1))
        for i in range(1, len_of_w):  # w[i]就是一个local_weights eg.len_of_w==4-->4个有效participants
            w_avg[k] += w[i][k]     # 需要优化——考虑datasize的比例
        # print("after for loop w_avg[k]: ", type(w_avg[k]))
        w_avg[k] = torch.div(w_avg[k], len_of_w)
    return w_avg


def FedAvg(w):
    len_of_w = len(w)
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len_of_w):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len_of_w)
    return w_avg
