'''
cifar non-iid
simulation_results/Jan21_V1/sim_res_2022V2/acc_cifar_non-iid_cnn_2022V2.png
simulation_results/Jan21_V2.1_TenGroupEachSelectOne/sim_res_2022V2/acc_cifar_non-iid_cnn_2022V2.png
simulation_results/Jan22_V3.0/sim_res_2022V3.0_cifar_niid_cnn/acc_cifar_non-iid_cnn_2022V3.0.png

cifar iid

mnist non-iid
simulation_results/Jan21_V2.1_TenGroupEachSelectOne/sim_res_2022V1/acc_mnist_non-iid_cnn_2022V1.png
simulation_results/Jan22_V3.0/sim_res_2022V3.0_mnist_niid_cnn/acc_mnist_non-iid_cnn_2022V3.0.png
res/myTempPlot/Jan23_1252_mnist_niid_cnn_2022V3_reward_mode_change_ruixin/acc_mnist_niid_cnn.png

mnist iid
simulation_results/Jan22_V3.0/sim_res_2022V3.0_mnist_iid_cnn/acc_mnist_iid_cnn_2022V3.0.png

'''


import numpy as np
import matplotlib.pyplot as plt

###################################################################################################
################################    V1        ####################################################
# acc_linucb_mnist_cnn_E200_C0.1_iid_False
# acc_random = np.loadtxt('res/myTempPlot/Jan23_1352_mnist_niid_cnn_2022V3_reward_mode_change_ruixin/acc_random_mnist_cnn_E200_C0.1_iid_False.txt')
# acc_fedcs = np.loadtxt('res/myTempPlot/Jan23_1352_mnist_niid_cnn_2022V3_reward_mode_change_ruixin/acc_fedcs_mnist_cnn_E200_C0.1_iid_False.txt')
# acc_ucb = np.loadtxt('res/myTempPlot/Jan23_1352_mnist_niid_cnn_2022V3_reward_mode_change_ruixin/acc_ucb_mnist_cnn_E200_C0.1_iid_False.txt')
# acc_linucb = np.loadtxt('res/myTempPlot/Jan23_1352_mnist_niid_cnn_2022V3_reward_mode_change_ruixin/acc_linucb_mnist_cnn_E200_C0.1_iid_False.txt')
###################################################################################################
###################################################################################################

###################################################################################################
################################    V2        #################################################
# acc_linucb_cifar_cnn_E200_C0.1_iid_False
# acc_random = np.loadtxt('res/cifar_iid/acc_random_cifar_cnn_E200_C0.1_iid_False.txt')
# acc_fedcs = np.loadtxt('res/cifar_iid/acc_fedcs_cifar_cnn_E200_C0.1_iid_False.txt')
# acc_ucb = np.loadtxt('res/cifar_iid/acc_ucb_cifar_cnn_E200_C0.1_iid_False.txt')
# acc_linucb = np.loadtxt('res/cifar_iid/acc_linucb_cifar_cnn_E200_C0.1_iid_False.txt')
###################################################################################################
###################################################################################################


###################################################################################################
##############################    Jan23 V3        ######################################################
# acc_linucb_mnist_cnn_E200_C0.1_iid_False
# acc_random = np.loadtxt('res/myTempPlot/Jan23_select20_than12_all_change_fedavg/mnist_niid_cnn/acc_random_mnist_cnn_E200_C0.1_iid_False.txt')
# acc_fedcs = np.loadtxt('res/myTempPlot/Jan23_select20_than12_all_change_fedavg/mnist_niid_cnn/acc_fedcs_mnist_cnn_E200_C0.1_iid_False.txt')
# acc_ucb = np.loadtxt('res/myTempPlot/Jan23_select20_than12_all_change_fedavg/mnist_niid_cnn/acc_ucb_mnist_cnn_E200_C0.1_iid_False.txt')
# acc_linucb = np.loadtxt('res/myTempPlot/Jan23_select20_than12_all_change_fedavg/mnist_niid_cnn/acc_linucb_mnist_cnn_E200_C0.1_iid_False.txt')

# acc_linucb_cifar_cnn_E200_C0.1_iid_True
# acc_random = np.loadtxt('res/myTempPlot/Jan23_select20_than12_all/cifar_iid_cnn/acc_random_cifar_cnn_E200_C0.1_iid_True.txt')
# acc_fedcs = np.loadtxt('res/myTempPlot/Jan23_select20_than12_all/cifar_iid_cnn/acc_fedcs_cifar_cnn_E200_C0.1_iid_True.txt')
# acc_ucb = np.loadtxt('res/myTempPlot/Jan23_select20_than12_all/cifar_iid_cnn/acc_ucb_cifar_cnn_E200_C0.1_iid_True.txt')
# acc_linucb = np.loadtxt('res/myTempPlot/Jan23_select20_than12_all/cifar_iid_cnn/acc_linucb_cifar_cnn_E200_C0.1_iid_True.txt')

# acc_linucb_cifar_cnn_E200_C0.1_iid_True
# acc_random = np.loadtxt('res/myTempPlot/Jan23_select20_than12_all/cifar_niid_resnet/acc_random_cifar_resnet_E200_C0.1_iid_False.txt')
# acc_fedcs = np.loadtxt('res/myTempPlot/Jan23_select20_than12_all/cifar_niid_resnet/acc_fedcs_cifar_resnet_E200_C0.1_iid_False.txt')
# acc_ucb = np.loadtxt('res/myTempPlot/Jan23_select20_than12_all/cifar_niid_resnet/acc_ucb_cifar_resnet_E200_C0.1_iid_False.txt')
# acc_linucb = np.loadtxt('res/myTempPlot/Jan23_select20_than12_all/cifar_niid_resnet/acc_linucb_cifar_resnet_E200_C0.1_iid_False.txt')


# acc_random = np.loadtxt('res/myTempPlot/Jan23_select20_than12_all_change_fedavg/cifar_niid_cnn/acc_random_cifar_cnn_E200_C0.1_iid_False.txt')
# acc_fedcs = np.loadtxt('res/myTempPlot/Jan23_select20_than12_all_change_fedavg/cifar_niid_cnn/acc_fedcs_cifar_cnn_E200_C0.1_iid_False.txt')
# acc_ucb = np.loadtxt('res/myTempPlot/Jan23_select20_than12_all_change_fedavg/cifar_niid_cnn/acc_ucb_cifar_cnn_E200_C0.1_iid_False.txt')
# acc_linucb = np.loadtxt('res/myTempPlot/Jan23_select20_than12_all_change_fedavg/cifar_niid_cnn/acc_linucb_cifar_cnn_E200_C0.1_iid_False.txt')


acc_random = np.loadtxt('res/myTempPlot/Jan23_select20_than12_all_change_fedavg/cifar_niid_resnet/acc_random_cifar_resnet_E200_C0.1_iid_False.txt')
acc_fedcs = np.loadtxt('res/myTempPlot/Jan23_select20_than12_all_change_fedavg/cifar_niid_resnet/acc_fedcs_cifar_resnet_E200_C0.1_iid_False.txt')
acc_ucb = np.loadtxt('res/myTempPlot/Jan23_select20_than12_all_change_fedavg/cifar_niid_resnet/acc_ucb_cifar_resnet_E200_C0.1_iid_False.txt')
acc_linucb = np.loadtxt('res/myTempPlot/Jan23_select20_than12_all_change_fedavg/cifar_niid_resnet/acc_linucb_cifar_resnet_E200_C0.1_iid_False.txt')

# acc_linucb_mnist_cnn_E200_C0.1_iid_True
# acc_random  = np.loadtxt('res/myTempPlot/Jan24_select10_than12_all_change_fedavg/mnist_iid_cnn/acc_random_mnist_cnn_E200_C0.1_iid_True.txt')
# acc_fedcs   = np.loadtxt('res/myTempPlot/Jan24_select10_than12_all_change_fedavg/mnist_iid_cnn/acc_fedcs_mnist_cnn_E200_C0.1_iid_True.txt')
# acc_ucb     = np.loadtxt('res/myTempPlot/Jan24_select10_than12_all_change_fedavg/mnist_iid_cnn/acc_ucb_mnist_cnn_E200_C0.1_iid_True.txt')
# acc_linucb  = np.loadtxt('res/myTempPlot/Jan24_select10_than12_all_change_fedavg/mnist_iid_cnn/acc_linucb_mnist_cnn_E200_C0.1_iid_True.txt')
###################################################################################################
###################################################################################################


###################################################################################################
#################################    V4        ######################################################
# acc_linucb_cifar_cnn_E200_C0.1_iid_True
# acc_random = np.loadtxt('res/cifar_iid/acc_random_cifar_cnn_E200_C0.1_iid_True.txt')
# acc_fedcs = np.loadtxt('res/cifar_iid/acc_fedcs_cifar_cnn_E200_C0.1_iid_True.txt')
# acc_ucb = np.loadtxt('res/cifar_iid/acc_ucb_cifar_cnn_E200_C0.1_iid_True.txt')
# acc_linucb = np.loadtxt('res/cifar_iid/acc_linucb_cifar_cnn_E200_C0.1_iid_True.txt')
###################################################################################################
###################################################################################################


###################################################################################################
####################################   new   #################################################
# acc_linucb_cifar_resnet_E200_C0.1_iid_True
# acc_random = np.loadtxt('res/myTempPlot/acc_random_cifar_resnet_E200_C0.1_iid_True.txt')
# acc_fedcs = np.loadtxt('res/myTempPlot/acc_fedcs_cifar_resnet_E200_C0.1_iid_True.txt')
# acc_ucb = np.loadtxt('res/myTempPlot/acc_ucb_cifar_resnet_E200_C0.1_iid_True.txt')
# acc_linucb = np.loadtxt('res/myTempPlot/acc_linucb_cifar_resnet_E200_C0.1_iid_True.txt')

# acc_linucb_cifar_resnet_E200_C0.1_iid_False
# acc_random = np.loadtxt('res/myTempPlot/acc_random_cifar_resnet_E200_C0.1_iid_False.txt')
# acc_fedcs = np.loadtxt('res/myTempPlot/acc_fedcs_cifar_resnet_E200_C0.1_iid_False.txt')
# acc_ucb = np.loadtxt('res/myTempPlot/acc_ucb_cifar_resnet_E200_C0.1_iid_False.txt')
# acc_linucb = np.loadtxt('res/myTempPlot/acc_linucb_cifar_resnet_E200_C0.1_iid_False.txt')



##################分割线--------------------------------------

###################################################################################################
###################################     V5   ####################################################
# acc_linucb_mnist_mlp_E200_C0.1_iid_False
# acc_random = np.loadtxt('res/cifar_iid/acc_random_mnist_mlp_E200_C0.1_iid_False.txt')
# acc_fedcs = np.loadtxt('res/cifar_iid/acc_fedcs_mnist_mlp_E200_C0.1_iid_False.txt')
# acc_ucb = np.loadtxt('res/cifar_iid/acc_ucb_mnist_mlp_E200_C0.1_iid_False.txt')
# acc_linucb = np.loadtxt('res/cifar_iid/acc_linucb_mnist_mlp_E200_C0.1_iid_False.txt')
###################################################################################################
###################################################################################################


###################################################################################################
###################################     V6   ####################################################
# acc_linucb_cifar_mlp_E200_C0.1_iid_False
# acc_random = np.loadtxt('res/cifar_iid/acc_random_cifar_mlp_E200_C0.1_iid_False.txt')
# acc_fedcs = np.loadtxt('res/cifar_iid/acc_fedcs_cifar_mlp_E200_C0.1_iid_False.txt')
# acc_ucb = np.loadtxt('res/cifar_iid/acc_ucb_cifar_mlp_E200_C0.1_iid_False.txt')
# acc_linucb = np.loadtxt('res/cifar_iid/acc_linucb_cifar_mlp_E200_C0.1_iid_False.txt')
###################################################################################################
###################################################################################################


# acc_random = np.loadtxt('res/mnist_noniid/acc_random_mnist_cnn_E200_C0.1_iid_False.txt')
# acc_fedcs = np.loadtxt('res/mnist_noniid/acc_fedcs_mnist_cnn_E200_C0.1_iid_False.txt')
# acc_ucb = np.loadtxt('res/mnist_noniid/acc_ucb_mnist_cnn_E200_C0.1_iid_False.txt')
# acc_linucb = np.loadtxt('res/mnist_noniid/acc_linucb_mnist_cnn_E200_C0.1_iid_False.txt')

# acc_random = np.loadtxt('res/cifar_iid/acc_random_cifar_cnn_E200_C0.1_iid_True.txt')
# acc_fedcs = np.loadtxt('res/cifar_iid/acc_fedcs_cifar_cnn_E200_C0.1_iid_True.txt')
# acc_ucb = np.loadtxt('res/cifar_iid/acc_ucb_cifar_cnn_E200_C0.1_iid_True.txt')
# acc_linucb = np.loadtxt('res/cifar_iid/acc_linucb_cifar_cnn_E200_C0.1_iid_True.txt')

'''
acc_random = np.loadtxt('res/cifar_noniid/acc_random_cifar_cnn_E200_C0.1_iid_False.txt')
acc_fedcs = np.loadtxt('res/cifar_noniid/acc_fedcs_cifar_cnn_E200_C0.1_iid_False.txt')
acc_ucb = np.loadtxt('res/cifar_noniid/acc_ucb_cifar_cnn_E200_C0.1_iid_False.txt')
acc_linucb = np.loadtxt('res/cifar_noniid/acc_linucb_cifar_cnn_E200_C0.1_iid_False.txt')
'''

# acc_random = np.loadtxt('res/mnist_noniid/acc_random_mnist_cnn_E100_C0.1_iid_False.txt')
# acc_fedcs = np.loadtxt('res/mnist_noniid/acc_fedcs_mnist_cnn_E100_C0.1_iid_False.txt')
# acc_ucb = np.loadtxt('res/mnist_noniid/acc_ucb_mnist_cnn_E100_C0.1_iid_False.txt')
# acc_linucb = np.loadtxt('res/mnist_noniid/acc_linucb_mnist_cnn_E100_C0.1_iid_False.txt')

# acc_random = np.loadtxt('res/mnist_iid/acc_random_mnist_cnn_E200_C0.1_iid_True.txt')
# acc_fedcs = np.loadtxt('res/mnist_iid/acc_fedcs_mnist_cnn_E200_C0.1_iid_True.txt')
# acc_ucb = np.loadtxt('res/mnist_iid/acc_ucb_mnist_cnn_E200_C0.1_iid_True.txt')
# acc_linucb = np.loadtxt('res/mnist_iid/acc_linucb_mnist_cnn_E200_C0.1_iid_True.txt')

# plt.plot(range(len(acc_random)), acc_random)
# plt.plot(range(len(acc_fedcs)), acc_fedcs)
# plt.plot(range(len(acc_ucb)), acc_ucb)
# plt.xlabel('round')
# plt.ylabel('Acc')
# plt.show()

acc_random_list = []
acc_fedcs_list  = []
acc_ucb_list    = []
acc_linucb_list = []

min_list        = []
max_list        = []

length = 200
interval = 7

for i in range(length):
    if i % interval == 0:
        acc_random_i = acc_random[i]
        acc_fedcs_i  = acc_fedcs[i]
        acc_ucb_i    = acc_ucb[i]
        acc_linucb_i = acc_linucb[i]

        # 存储数据
        acc_random_list.append(acc_random_i)
        acc_fedcs_list.append(acc_fedcs_i)
        acc_ucb_list.append(acc_ucb_i)
        acc_linucb_list.append(acc_linucb_i)

        # 找到最大最小值
        compare_list = [acc_random_i, acc_fedcs_i, acc_ucb_i]
        min_list.append(min(compare_list))
        max_list.append(max(compare_list))


round = range(length)
rount_interval = np.arange(0, length, interval)

# plt.title('mnist noniid prob: 0.8')
# plt.plot(round, acc_random[:length], label='random')
# plt.plot(round, acc_fedcs[:length], label='fedcs')
# plt.plot(round, acc_ucb[:length], label='ucb')
# plt.plot(round, acc_linucb[:length], label='linucb')

# plt.plot(rount_interval, acc_random_list, 'v-', label='random')
# plt.plot(rount_interval, acc_fedcs_list, '*-', label='fedcs')
# plt.plot(rount_interval, acc_ucb_list, '+--', label='ucb')
plt.xticks([0, 50, 100, 150, 200])
plt.plot(rount_interval, acc_linucb_list, 'v--', linewidth=1.5, label='linucb')
plt.fill_between(rount_interval, min_list, max_list, color='#ff7f0e', alpha=0.25)
plt.tick_params(labelsize=18)
plt.xlabel('FL Rounds')
plt.ylabel('Test Accuracy')
# plt.ylim(0, 60)
plt.legend()

import configuration as conf

# fig_savepath = conf.FL_PATH + 'simulation/simulation_results/Jan23V3.2_20chosen_than12_server39/sim_res_2022V3.2_cifar_iid_resnet/acc_cifar_iid_resnet_2022V3.2.png'  #Jan22_V3.2
# fig_savepath = conf.FL_PATH + 'simulation/simulation_results/Jan23V3.2_20chosen_than12_server39/sim_res_2022V3.2_cifar_niid_resnet/acc_cifar_niid_resnet_2022V3.2.png'
# fig_savepath = conf.FL_PATH + 'simulation/simulation_results/Jan22V3.2_20chosen_than12_server37/sim_res_2022V3.2_mnist_niid_cnn/acc_mnist_non-iid_cnn_2022V3.2.png'  #Jan22_V3.2
# fig_savepath = conf.FL_PATH + 'simulation/simulation_results/Jan22V3.2_20chosen_than12_server37/sim_res_2022V3.2_mnist_iid_cnn/acc_mnist_iid_cnn_2022V3.2.png'
# fig_savepath = conf.FL_PATH + 'simulation/simulation_results/Jan22_V3.0/sim_res_2022V3.0_mnist_iid_cnn/acc_mnist_iid_cnn_2022V3.0.png'
# fig_savepath = conf.FL_PATH + 'simulation/simulation_results/Jan23V3.2_20chosen_than12_localcomputer/sim_res_2022V3.2_cifar_niid_resnet_small/acc_cifar_non-iid_resnet_small_2022V3.2.png'
# fig_savepath = conf.FL_PATH + 'simulation/simulation_results/Jan23V3.2_20chosen_than12_localcomputer/sim_res_2022V3.2_cifar_iid_resnet_small/acc_cifar_non-iid_resnet_small_2022V3.2.png'

# fig_savepath = conf.FL_PATH + 'simulation/simulation_results/sim_res_2022V5/acc_mnist_non-iid_mlp_2022V5.png'
# fig_savepath = conf.FL_PATH + 'simulation/simulation_results/sim_res_2022V6/acc_cifar_non-iid_mlp_2022V6.png'
# fig_savepath = conf.FL_PATH + 'simulation/simulation_results/sim_res_2022V7/acc_mnist_iid_mlp_2022V7.png'
# fig_savepath = conf.FL_PATH + 'simulation/simulation_results/sim_res_2022V8/acc_cifar_iid_mlp_2022V8.png'
# fig_savepath = conf.FL_PATH + 'simulation/res/myTempPlot/Jan23_1352_mnist_niid_cnn_2022V3_reward_mode_change_ruixin/acc_mnist_niid_cnn.png'


# fig_savepath = conf.FL_PATH + 'simulation/res/myTempPlot/Jan23_select20_than12_all/cifar_niid_cnn/figure/acc_cifar_niid_cnn.png'
# fig_savepath = conf.FL_PATH + 'simulation/res/myTempPlot/Jan24_select10_than12_all_change_fedavg/mnist_iid_cnn/figure/acc_mnist_iid_cnn.png'
# fig_savepath = conf.FL_PATH + 'simulation/res/myTempPlot/Jan23_select20_than12_all_change_fedavg/cifar_niid_resnet/figure/acc_cifar_niid_resnet.png'
# fig_savepath = conf.FL_PATH + 'simulation/res/myTempPlot/Jan23_select20_than12_all_change_fedavg/mnist_iid_cnn/figure/acc_mnist_iid_cnn.png'
fig_savepath = conf.FL_PATH + 'simulation/res/myTempPlot/Jan23_select20_than12_all_change_fedavg/cifar_niid_resnet/figure/acc_cifar_niid_resnet_with_interval_fillColor.png'
plt.savefig(fig_savepath)   # 新增 Jan 18

plt.show()
