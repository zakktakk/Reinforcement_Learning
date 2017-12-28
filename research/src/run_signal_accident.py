#-*- coding: utf-8 -*-
# author : Takuro Yamazaki
# description : シグナリング＋事故

import os
import numpy as np
import pandas as pd
from collections import OrderedDict

"""simulation world"""
from world import synchro_world_signal

"""network"""
from networks import network_utils

"""agent"""
from agent import Q_Learning_Agent as ql

"""payoff matrix"""
from world.payoff_matrix import *


# graphの定義
pcG = network_utils.graph_generator.powerlaw_cluster_graph

all_after = [pd.DataFrame(np.array([[3, 3, 0, 0],[3, 3, 0, 0],[5, 5, 0, 0],[5, 5, 0, 0]]),
                    index=['cs', 'c', 'ds', 'd'], columns=['cs', 'c', 'ds', 'd']),
             pd.DataFrame(np.array([[30, 30, 10, 10], [30, 30, 10, 10], [5, 5, 1, 1], [5, 5, 1, 1]]),
                          index=['cs', 'c', 'ds', 'd'], columns=['cs', 'c', 'ds', 'd']),
             pd.DataFrame(np.array([[4, 4, 0, 0], [4, 4, 0, 0], [0, 0, 2, 2], [0, 0, 2, 2]]),
                          index=['cs', 'c', 'ds', 'd'], columns=['cs', 'c', 'ds', 'd']),
            ]


# share rateの定義
all_p = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]


RESULT_DIR = "../results/signal_accident/powerlaw_cluster/q"
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

for ap in all_p:
    for ti, aa in zip(["kaishou", "kakudai", "coodinate"], all_after):
        for k in range(5):
            RESULT_NAME = RESULT_DIR+"/"+str(k)+"/prisoner"+"_"+str(ap*100)+"_"+ti
            if not os.path.exists("/".join(RESULT_NAME.split("/"))[:-1]):
                os.makedirs("/".join(RESULT_NAME.split("/"))[:-1])
            W = synchro_world_signal.synchro_world_signal(100, 1000, prisoners_dilemma_sig(), pcG, ql.Q_Learning_Agent,
                                                          p_noise=ap, altered_mat=aa)
            W.run()
            W.save(RESULT_NAME)

print('done!!')
