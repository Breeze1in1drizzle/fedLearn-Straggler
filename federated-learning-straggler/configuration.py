# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 18:09:02 2021
@author(s): HuangLab

description:
the configuration (including paths) for the whole project
"""

import os
abs_path = os.path.abspath(__file__)
# print(abs_path)

# this variable is the root path in edgeCSFproject
ROOT_PATH = abs_path.replace('\\', '/')[:-16]
'''
print("ROOT_PATH:\n", ROOT_PATH)
result of the above line of codes
ROOT_PATH:
 D:/myworkspace/github_workspace/ClientSelectionFed-Breeze/edgeCSFproject/
'''
ALGORITHM_PATH = ROOT_PATH + 'algorithm/'
PLOT_FIGS_PATH = ROOT_PATH + 'PlotFigs/'

# this variable decide where the dataset is
DATASET_PATH = ROOT_PATH + 'dataset/'
'''
print("DATASET_PATH:\n", DATASET_PATH)
result of the above line of code
DATASET_PATH:
 D:/myworkspace/github_workspace/ClientSelectionFed-Breeze/edgeCSFproject/dataset/
'''
# this variable decide where the model is saved
MODEL_PATH = ROOT_PATH + 'model/'


FL_PATH = ROOT_PATH + 'fedLearnSim/'    # federated learning simulation

'''
#########################################################################
######################## to be defined ##################################
#########################################################################
'''

# this variable decide where the data collected by data collection framework is saved
COLLECTED_DATA_PATH = DATASET_PATH + 'collected_dataset/'   # to be defined
# this variable describes the port which this project will take (in some cases)
PORT = 5000     # to be defined
