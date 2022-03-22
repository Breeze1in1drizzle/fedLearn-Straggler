'''
这个文件用于模拟时间消耗（各设备的时间消耗）
这里用到指数分布
指数分布指的是时间间隔——一段时间内所有clients完成训练的概率的分布
如果是cross-device federated learning，那么使用指数分布也是合理的
但是如果是节点不多的cross-silo federated learning，那么指数分布未必合理？
'''

import math
import numpy as np
import matplotlib.pyplot as plt


def time_test():
    '''
    设定好各个clients的时间模拟的思路与框架
    也可以考虑把这些时间输出成为一个npy文件，之后调用的时候都是按照这个来
    '''
    total_round = 200   # 200个communication round
    clients = []
    tau = 70
    for i in range(10):     # 10个clients
        # 这个分布是指当前轮次communication round中，某个设备训练一个epoch需要的时间的集合
        client = np.random.exponential(tau, size=10)
        clients.append(client)
        print("client ", i)
        print(client)
        # print(client[0])


def test_main_v1():
    '''
    https://blog.csdn.net/qq_26948675/article/details/79589633
    :return:
    '''

    def successive_poisson(tau1, tau2, size=1):
        # Draw samples out of first exponential distribution: t1
        t1 = np.random.exponential(tau1, size=size)
        # Draw samples out of second exponential distribution: t2
        t2 = np.random.exponential(tau2, size=size)
        return t1 + t2
        # return t1

    # Draw samples of waiting times
    waiting_times = successive_poisson(764, 715, size=100000)
    # waiting_times = successive_poisson(50, 70, size=100000)
    # Make the histogram
    _ = plt.hist(waiting_times, bins=100, histtype='step')  #, normed=True)
    # Label axes
    _ = plt.xlabel('total waiting time (games)')
    _ = plt.ylabel('PDF')
    # Show the plot
    plt.show()


def test_one_client(datasize=100, a_k=0.01):
    '''
    a_k==0.01，即1/100，也就是1min处理100条数
    '''
    local_epoch = 5
    # miu_k = local_epoch * datasize * 0.7        # 将lamda控制在 1/2 这个数值上，不要让execution time的分布太集中
    miu_k = local_epoch * datasize * 0.5        # 将lamda控制在 1/2 这个数值上，不要让execution time的分布太集中
    # miu_k = 1 / a_k
    # miu_k = 400

    client1 = client(datasize=datasize, a_k=a_k, miu_k=miu_k, local_epoch=local_epoch)

    lamda = client1.get_lamda()
    print(lamda)
    # print(math.log(2, math.e) / lamda)  # 以e为底，2的对数-->ln2
    exec_time = client1.get_execution_time(round=200)        # 模拟200 rounds期间，每一次round所消耗的时间
    print(exec_time)
    print(np.var(exec_time))


def test_multiple_clients():
    '''
    测试3个clients
    （其实这里的clients泛指一个节点的计算能力，也可以指organization or devices）
    '''
    test_one_client(datasize=800, a_k=0.001)
    # test_one_client(datasize=600, a_k=0.02)
    # test_one_client(datasize=400, a_k=0.005)
    test_one_client(datasize=850, a_k=0.001)
    test_one_client(datasize=900, a_k=0.001)


class client():
    '''
    设定是一共10个client，可以视作cross-silo形式
    10个client，随机出datasize出来，然后固定各client的datasize
    再用这些固定的datasize去切分数据集non-iid形式
    10类图片，每一类6000张
    分给10个client，每个client一个main label
    每个client 600张
    可以考虑设置多一点，设置800为basic value，然后再设置一个random让数值从800-900之间变换
    a_k-->计算能力，单位时间处理的数据条数
    '''
    def __init__(self, datasize, a_k, miu_k, local_epoch):
        self.a_k    = a_k                       # 表示计算能力的系数，乘以datasize从而表示处理完一次本地数据需要消耗的时间
        self.miu_k  = miu_k                     # 一个系数，除以(local_epoch*datasize)从而表示指数分布的lamda
        self.local_epoch    = local_epoch       # 本地训练的epoch
        self.excution_time  = 0                 # 当前轮次设备的执行时间（初始化为）
        self.datasize       = datasize          # 该设备的数据集大小（表示有多少条数据）

    def set_coefficient(self, datasize, a_k, miu_k, local_epoch):
        self.a_k    = a_k                   # 表示计算能力的系数，乘以datasize从而表示处理完一次本地数据需要消耗的时间
        self.miu_k  = miu_k                 # 一个系数，除以(local_epoch*datasize)从而表示指数分布的lamda
        self.local_epoch    = local_epoch   # 本地训练的epoch
        self.excution_time  = 0             # 当前轮次设备的执行时间（初始化为）
        self.datasize       = datasize      # 该设备的数据集大小（表示有多少条数据）

    def get_lamda(self):
        return (self.miu_k / (self.local_epoch * self.datasize))

    def get_execution_time(self, round=1):
        '''
        获取一个设备的所有round的执行时间
        round指的是一共运行多少个FL communication round
        '''
        lamda = self.get_lamda()
        temp_time = np.random.exponential(1/lamda, size=round)       # 生成一个temp_time = execution_time - local_epoch * a_k * datasize
        execution_time = temp_time + self.local_epoch * self.a_k * self.datasize
        return execution_time


if __name__ == "__main__":
    # time_test()
    # test_main_v1()
    test_multiple_clients()
    # pass
