#-*- coding: utf-8 -*-
# author : Takuro Yamazaki
# description : 事故が起こった場合の状況、グラフ構造ごとの学習の違い

import os
import itertools
import pandas as pd
import numpy as np
from collections import OrderedDict

"""simulation world"""
from world import synchro_world

"""network"""
from networks import network_utils

"""agent"""
# defaultはこの4種類にしよう
from agent import Q_Learning_Agent as ql

"""payoff matrix"""
from world.payoff_matrix import *


graph_prefix = "../src/networks/"

# graphの定義
cG = network_utils.graph_generator.complete_graph
pcG = network_utils.graph_generator.powerlaw_cluster_graph

all_graph = OrderedDict((("powerlaw_cluster",pcG), ("complete",cG),
                         ("multiple_clustered",graph_prefix+"multiple_clustered.gpickle"),
                         ("inner_dence_clustered", graph_prefix+"inner_dence_clustered.gpickle"),
                         ("one_dim_regular", graph_prefix+"onedim_regular.gpickle")))


# accident後の行列
all_after = [pd.DataFrame(np.array([[3, 0],[2, 0]]),index=list('cd'), columns=list('cd')),
             pd.DataFrame(np.array([[30, 10], [5, 1]]), index=list('cd'), columns=list('cd')),
             pd.DataFrame(np.array([[4, 0], [0, 2]]), index=list('cd'), columns=list('cd'))
             ]


RESULT_DIR = "../results/accident_graph/q/"
for G in all_graph:
    if not os.path.exists(RESULT_DIR+"/"+G):
        os.makedirs(RESULT_DIR+"/"+G)

    print("  "+G)
    for ti, aa in zip(["kaishou", "kakudai", "coodinate"], all_after):
        for k in range(5):
            RESULT_NAME = RESULT_DIR+"/"+G+"/"+str(k)+"/prisoners_dilemma_"+ti
            if not os.path.exists("/".join(RESULT_NAME.split("/"))[:-1]):
                os.makedirs("/".join(RESULT_NAME.split("/"))[:-1])
            W = synchro_world.synchro_world(100, 1000, prisoners_dilemma(), all_graph[G], ql.Q_Learning_Agent, altered_mat=aa)
            W.run()
            W.save(RESULT_NAME)

print('done!!')
