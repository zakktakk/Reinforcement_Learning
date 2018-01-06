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
from agent import Actor_Critic_Agent as aca

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
all_after = [[2,2,2,-2], [2,-10,2,10], [10,10,10,-10], [10,-10,10,10]]


RESULT_DIR = "../results/accident_graph/"
for alg_name, alg in zip(["q", "sarsa"], [ql.Q_Learning_Agent, aca.Actor_Critic_Agent]):
    for G in all_graph:
        print("  "+G)

        for ti, aa in zip(["reverse", "kakusa", "big_reverse", "infration"], all_after):
            for k in range(5):
                RESULT_NAME = RESULT_DIR+G+"/"+alg_name+"/"+str(k)+"/nipd_"+ti

                if not os.path.exists("/".join(RESULT_NAME.split("/")[:-1])):
                    os.makedirs("/".join(RESULT_NAME.split("/")[:-1]))

                W = synchro_world.synchro_world(100, 1000, [2,-2,2,2], all_graph[G], alg, altered_mat=aa)
                W.run()
                W.save(RESULT_NAME)

print('done!!')
