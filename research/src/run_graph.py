#-*- coding: utf-8 -*-
# author : Takuro Yamazaki
# description : グラフ構造ごとの協調行動の違い

import os
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

RESULT_DIR = "../results/graph/q/"
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

for G in all_graph:
    if not os.path.exists(RESULT_DIR+"/"+G):
        os.makedirs(RESULT_DIR+"/"+G)
    print("  "+G)
    for k in range(5):
        RESULT_NAME = RESULT_DIR+"/"+G+"/"+str(k)+"/prisoners_dilemma"
        W = synchro_world.synchro_world(100, 1000, prisoners_dilemma(), all_graph[G], ql.Q_Learning_Agent)
        W.run()
        W.save(RESULT_NAME)

print('done!!')
