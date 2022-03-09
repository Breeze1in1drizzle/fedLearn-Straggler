'''
Created on March 9 2022 14:52:43
@author(s): HuangLab

今天需要完成的是把这些代码精简化，然后确保能够跑通
需要注意，non-iid数据的拆分，依然可以考虑用高斯分布进行拆分
把sampling.py文件中的随机拆分的non-iid方法改成高斯分布拆分（并且固定一种拆分方式-->固定种子）

然后明天再把模拟时间的代码写完，就完成了初步的 FL simulation
'''

import matplotlib
# matplotlib.use('Agg')  # 绘图不显示

import configuration as conf
import matplotlib.pyplot as plt
import copy
import numpy as np
import pandas as pd

import torch
from torchvision import transforms, datasets

import time

import sys
import configuration as conf

# 把 sys path 添加需要 import 的文件夹
sys.path.append(conf.ROOT_PATH + 'fedLearn/')


from fedLearnSim.utils.sampling import mnist_iid, cifar_iid, mnist_noniid, cifar_noniid
from fedLearnSim.utils.sampling import mnist_iid_modified, cifar_iid_modified, mnist_noniid_modified
from fedLearnSim.utils.options import args_parser
from fedLearnSim.models.Update import LocalUpdate
from fedLearnSim.models.Nets import MLP, CNNMnist, CNNCifar, CNNCifarPlus
from fedLearnSim.models.resnet import ResNet
from fedLearnSim.models.Fed import FedAvg, FedAvgV1
from fedLearnSim.models.test import test_img


# settings
# mnist noniid:  --dataset mnist --num_channels 1 --model cnn --epochs 200 --gpu 0
# cifar iid: --dataset cifar --iid --num_channels 3 --model cnn --epochs 50 --gpu 0


def FedLearnSimulate(alg_str='linucb', args_model='cnn', valid_list_path="valid_list_linucb.txt",
                     args_dataset='mnist', args_usernumber=57, args_iid=False, map_file=None):
    '''
    这个函数是执行 Federated Learning Simulation 的 main 函数
    这里我需要做的应该主要是2个内容：
    ①划分dataset（iid & non-iid）跑 mnist & cifar；（似乎有划分的代码了，我只需要看懂）
    ②理解valid_list_linucb
    '''
    # 保存文本的时候使用，此时是 LinUCB 方法
    result_str = alg_str

    # valid_list = np.loadtxt('noniid_valid/valid_list_fedcs.txt')
    # valid_list = np.loadtxt('valid_list_linucb.txt')                # 这个是 iid 情况的 devices selection 文件
    valid_list = np.loadtxt(valid_list_path, encoding='utf_8_sig')
    # 10*200 --> 猜测应该是 200 communication rounds，10个设备中每行（即每个round）设备数值不为-1的就可以挑选

    # load args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    # load args

    # 调参
    args.local_bs = 20
    args.local_ep = 10

    print("cuda is available : ", torch.cuda.is_available())        # 本次实验使用的 GPU 型号为 RTX 2060 SUPER，内存专用8G、共享8G




    print("load dataset")
    #####################################################################################################################
    #####################################################################################################################
    # load dataset
    args.dataset    = args_dataset
    args.num_users  = args_usernumber
    args.iid        = args_iid                 # non-iid
    if args.dataset == 'mnist':
        print("mnist dataset!")
        trans_mnist     = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train   = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test    = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users (100)
        # args.iid = True
        if args.iid:        # 好像没看见 non-iid 的代码
            print("args.iid is true")
            # dict_users = mnist_iid(dataset_train, args.num_users)
            dict_users = mnist_iid_modified(dataset_train, args.num_users)
        else:
            print("args.iid is false, non-iid")
            # dict_users = mnist_noniid(dataset_train, args.num_users)
            dict_users = mnist_noniid_modified(dataset_train, args.num_users,
                                               main_label_prop=0.8, other=9, map_file=map_file)
            print("args.iid is false, non-iid")
    elif args.dataset == 'cifar':
        print("cifar dataset!")
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            print("cifar iid")
            # dict_users = cifar_iid(dataset_train, args.num_users)
            dict_users = cifar_iid_modified(dataset_train, args.num_users)
            # print("dict_users: ", type(dict_users))
            # print(dict_users)
        else:
            print("cifar non-iid")
            dict_users = cifar_noniid(dataset_train, args.num_users,
                                      min_train=200, max_train=1000, main_label_prop=0.8, other=9, map_file=map_file)
    else:
        exit('Error: unrecognized dataset')
    # load dataset
    #####################################################################################################################
    #####################################################################################################################





    img_size = dataset_train[0][0].shape




    print("build model")
    #####################################################################################################################
    #####################################################################################################################
    # build model
    args.model = args_model
    if args.model == 'cnn' and args.dataset == 'cifar':
        print("cnn & cifar")
        # global_net = CNNCifar(args=args).to(args.device)
        global_net = CNNCifarPlus(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        print("cnn & mnist")
        args.num_channels = 1
        global_net = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        print("mlp & cifar")
        len_in = 1
        for x in img_size:
            len_in *= x
        global_net = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    elif args.model == 'resnet' and args.dataset == 'cifar':
        global_net = ResNet(18, num_classes=10).to(args.device)
        print("resnet:\n", global_net)
    else:
        exit('Error: unrecognized model')
    # build model
    #####################################################################################################################
    #####################################################################################################################




    print("global_net:\n", global_net)

    global_net.train()

    # start time
    # time_start = time.time()

    # copy weights
    w_glob = global_net.state_dict()

    # 这几个最后保存为txt
    loss_avg_client  = []
    acc_global_model = []
    valid_number_list     = []

    #####################################################################################################################
    #####################################################################################################################
    # training
    if args.all_clients:
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]

    last_loss_avg = 0
    last_acc_global = 0


    args.epochs = 200           # 默认是10，现在设置成200，测试一下 cifar10 准确率能到多少——Jan 17 2022 23:19:50
    print("args.epochs: ", args.epochs)                  # default --> 10
    for round in range(args.epochs):
        loss_locals = []
        if not args.all_clients:
            w_locals = []
        round_idx = valid_list[round]       # valid_list 两百行 --> 200 round      即 round_idx是单独一行（one round）的数据（10个数值）
        user_idx_this_round = round_idx[np.where(round_idx != -1)]      # 一行数据中，等于-1的不选，其它的选上
        # print("user_idx_this_round:\n", user_idx_this_round)

        # 随机
        # user_idx_this_round = np.random.choice(range(args.num_users), 10, replace=False)  # 在num_users里面选m个

        # print("dict_user:\n", type(dict_users), '\n', dict_users)
        # total_data_sum = 0      # 所有设备datasize相加
        # for i in range(len(dict_users)):        # 每行dict_user[idx]的长度，就代表了数据量
        #     total_data_sum += len(dict_users[i])
        # valid_number_list.append(user_idx_this_round)   # 每轮valid participants 数量
        # print("user_idx_this_round:\n", user_idx_this_round)
        if len(user_idx_this_round) > 0:
            total_data_sum = 0  # 所有设备datasize相加
            for ix in user_idx_this_round:
                # print("ix: ", ix)
                total_data_sum += len(dict_users[ix])
            # print("total_data_sum: ", total_data_sum)
            for idx in user_idx_this_round:     # 遍历可选的设备
                # local updates?
                # print("dataset_train: ", dataset_train.shape)

                # print("dict_user[idx]: ", dict_users[idx])      # 每行[idx]的elements个数不定，一维的list

                local = LocalUpdate(args=args, dataset=dataset_train,
                                    idxs=dict_users[idx])
                weight, loss = local.train(net=copy.deepcopy(global_net).to(args.device))
                if args.all_clients:
                    w_locals[idx] = copy.deepcopy(weight)
                else:
                    w_locals.append(copy.deepcopy(weight))      # append    # 根据user_idx_this_round的顺序 append 上去
                loss_locals.append(copy.deepcopy(loss))

            # datasize = float(map_file.iloc[i, map_file.columns.get_loc('datasize')])
            # datasize_sum = float(map_file['datasize'].sum())
            # update global weights


            w_glob = FedAvgV1(w=w_locals, total_data_sum=total_data_sum,
                            user_idx_this_round=user_idx_this_round,
                            dict_users=dict_users)


            # w_glob = FedAvg(w=w_locals)   # 聚合模型得到全局模型

            # copy weight to net_glob
            global_net.load_state_dict(w_glob)
            # print loss
            loss_avg = sum(loss_locals) / len(loss_locals)
            loss_avg_client.append(loss_avg)
            acc_test, loss_test = test_img(global_net, dataset_test, args)
            acc_global_model.append(acc_test)
            last_loss_avg = loss_avg
            last_acc_global = acc_test
            print('Round {:3d}, Average loss {:.3f}, Global acc: {:.3f}, valid {:3d}'
                  .format(round, loss_avg, acc_test, len(user_idx_this_round)))
        else:

            print('Round {:3d}, Average loss {:.3f}, Global acc: {:.3f} 0 !'
                  .format(round, last_loss_avg, last_acc_global))
            loss_avg_client.append(last_loss_avg)
            acc_global_model.append(last_acc_global)
    # training
    #####################################################################################################################
    #####################################################################################################################


    # time_end = time.time()
    # print('totally cost time: {:3f}s'.format(time_end - time_start))

    # # plot loss curve
    # plt.figure()
    # plt.plot(range(len(loss_avg_client)), loss_avg_client)
    # plt.ylabel('train_loss')
    # plt.savefig('loss_random_{}_{}_E{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

    plt.figure()
    plt.plot(range(len(acc_global_model)), acc_global_model)
    plt.ylabel('acc_global')
    plt.show()
    # plt.savefig('acc_random_{}_{}_E{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

    np_valid_number_list = np.array(valid_number_list)
    # 保存loss和acc
    np.savetxt('res/cifar_iid/loss_{}_{}_{}_E{}_C{}_iid_{}.txt'.format(result_str, args.dataset, args.model,
                                                                       args.epochs, args.frac, args.iid), loss_avg_client)
    np.savetxt('res/cifar_iid/acc_{}_{}_{}_E{}_C{}_iid_{}.txt'.format(result_str, args.dataset, args.model, args.epochs,
                                                                      args.frac, args.iid), acc_global_model)
    # np.savetxt('res/cifar_iid/valid_{}_{}_{}_E{}_C{}_iid_{}.txt'.format(result_str, args.dataset, args.model, args.epochs,
    #                                                                   args.frac, args.iid), np_valid_number_list)


    # # testing
    # global_net.eval()
    # acc_train, loss_avg_client = test_img(global_net, dataset_train, args)
    # acc_test, loss_test = test_img(global_net, dataset_test, args)
    # print("Training accuracy: {:.2f}".format(acc_train))
    # print("Testing accuracy: {:.2f}".format(acc_test))


def multiSimulateMain():
    basicPath      = conf.FL_PATH

    folderPath_iid = basicPath + 'simulation/valid_list_data/2022Jan24_10chosen_ruixin_change_constrain/valid_iid/'# iid

    folderPath = basicPath + 'simulation/valid_list_data/2022Jan24_10chosen_ruixin_change_constrain/valid_niid/'

    # 映射表 设备id --> datasize
    # map_filePath = conf.DATASET_PATH + 'decision_making_dataset/MAP_ID2DataSize2022V1/map_id_to_datasize.csv'
    map_filePath = conf.DATASET_PATH + 'decision_making_dataset/MAP_ID2DataSize2022V2/map_id_to_datasize.csv'
    df_MAP_ID2DATASIZE = pd.DataFrame(pd.read_csv(map_filePath), index=None)
    args_usernumber = len(df_MAP_ID2DATASIZE)

    # 定义算法名字
    linucb_str = 'linucb'
    ucb_str = 'ucb'
    fedcs_str = 'fedcs'
    random_str = 'random'

    # 实验文本保存路径
    linucbPath = folderPath + 'valid_list_' + linucb_str + '.txt'
    ucbPath = folderPath + 'valid_list_' + ucb_str + '.txt'
    fedcsPath = folderPath + 'valid_list_' + fedcs_str + '.txt'
    randomPath = folderPath + 'valid_list_' + random_str + '.txt'

    linucbPath_iid = folderPath_iid + 'valid_list_' + linucb_str + '.txt'
    ucbPath_iid = folderPath_iid + 'valid_list_' + ucb_str + '.txt'
    fedcsPath_iid = folderPath_iid + 'valid_list_' + fedcs_str + '.txt'
    randomPath_iid = folderPath_iid + 'valid_list_' + random_str + '.txt'




    ########################################################################################################################
    ##########################     new         #####################################################################
    # non-iid cifar resnet
    FedLearnSimulate(args_dataset='cifar', args_usernumber=args_usernumber, args_model='resnet',
                     valid_list_path=linucbPath, alg_str=linucb_str, args_iid=False, map_file=df_MAP_ID2DATASIZE)
    FedLearnSimulate(args_dataset='cifar', args_usernumber=args_usernumber, args_model='resnet',
                     valid_list_path=fedcsPath, alg_str=fedcs_str, args_iid=False, map_file=df_MAP_ID2DATASIZE)
    FedLearnSimulate(args_dataset='cifar', args_usernumber=args_usernumber, args_model='resnet',
                     valid_list_path=randomPath, alg_str=random_str, args_iid=False, map_file=df_MAP_ID2DATASIZE)
    FedLearnSimulate(args_dataset='cifar', args_usernumber=args_usernumber, args_model='resnet',
                     valid_list_path=ucbPath, alg_str=ucb_str, args_iid=False, map_file=df_MAP_ID2DATASIZE)

    print("multi-simulation end")


if __name__ == '__main__':
    multiSimulateMain()


