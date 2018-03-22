#-*- coding: utf-8 -*-
# author : Takuro Yamazaki
# description : 全員の行動だけが観測可能な場合

import os
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

all_p = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

# graphの定義
cG = network_utils.graph_generator.complete_graph
pcG = network_utils.graph_generator.powerlaw_cluster_graph


all_graph = OrderedDict((("powerlaw_cluster",pcG), ("complete",cG),
                         ("multiple_clustered",graph_prefix+"multiple_clustered.gpickle"),
                         ("inner_dence_clustered", graph_prefix+"inner_dence_clustered.gpickle"),
                         ("one_dim_regular", graph_prefix+"onedim_regular.gpickle")))


RESULT_DIR = "../results/share_q_noise/"


for g in all_graph:
    for p in all_p:
        for k in range(5):
            RESULT_NAME = RESULT_DIR + g + "/q/" + str(k) + "/nipd_" + str(p * 100)

            if not os.path.exists("/".join(RESULT_NAME.split("/")[:-1])):
                os.makedirs("/".join(RESULT_NAME.split("/")[:-1]))

            W = synchro_world.synchro_world(100, 1000, [2, -2, 2, 2], all_graph[g], ql.Q_Learning_Agent, share_rate=p)
            W.run()
            W.save(RESULT_NAME)

print('done!!')
