#-*- coding: utf-8 -*-
# author : Takuro Yamazaki
# description : 実験2、グラフ構造とアルゴリズムごとの結果の違い

import os
import numpy as np
import pandas as pd
from collections import OrderedDict

"""simulation world"""
from world import synchro_world_observable

"""network"""
from networks import network_utils

"""agent"""
# defaultはこの4種類にしよう
from agent import Q_Learning_Agent as ql

"""payoff matrix"""
from world.payoff_matrix import *


# graphの定義
pcG = network_utils.graph_generator.powerlaw_cluster_graph

all_after = [pd.DataFrame(np.array([[3, 0],[5, 0]]),index=list('cd'), columns=list('cd')),
             pd.DataFrame(np.array([[30, 10], [5, 1]]), index=list('cd'), columns=list('cd')),
             pd.DataFrame(np.array([[4, 0], [0, 2]]), index=list('cd'), columns=list('cd'))
             ]

all_p = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]


RESULT_DIR = "../results/observable_accident/random/q/"
os.makedirs("../results/observable_accident/random/q")


for ti, aa in zip(["kaishou", "kakudai", "coodinate"], all_after):
    for ap in all_p:
        for k in range(5):
            RESULT_NAME = RESULT_DIR +str(k)+"/prisoners_dilemma_" + ti + "_" + str(ap*100)
            if not os.path.exists("/".join(RESULT_NAME.split("/")[:-1])):
                os.makedirs("/".join(RESULT_NAME.split("/")[:-1]))
            print(aa)
            W = synchro_world_observable.synchro_world_observable(100, 10000, prisoners_dilemma(), pcG, ql.Q_Learning_Agent,
                                            altered_mat=aa, p_noise=ap)
            W.run()
            W.save(RESULT_NAME)


print('done!!')
