#-*- coding: utf-8 -*-
# author : Takuro Yamazaki
# description : Q値を共有した時、事故が起きるシナリオ

import os
import numpy as np
import pandas as pd
from collections import OrderedDict

"""simulation world"""
from world import synchro_world

"""network"""
from networks import network_utils

"""agent"""
from agent import Q_Learning_Agent as ql
from agent import Actor_Critic_Agent as aca


"""payoff matrix"""
from world.payoff_matrix import *

graph_prefix = "../src/networks/"


# graphの定義
cG = network_utils.graph_generator.complete_graph

pcG = network_utils.graph_generator.powerlaw_cluster_graph

all_after = [[2,2,2,-2], [2,-10,2,10], [10,10,10,-10], [10,-10,10,10]]


all_graph = OrderedDict((("powerlaw_cluster",pcG), ("complete",cG),
                         ("multiple_clustered",graph_prefix+"multiple_clustered.gpickle"),
                         ("inner_dence_clustered", graph_prefix+"inner_dence_clustered.gpickle"),
                         ("one_dim_regular", graph_prefix+"onedim_regular.gpickle")))


RESULT_DIR = "../results/share_q_accident_graph/"

for alg_name, alg in zip(["q", "sarsa"], [ql.Q_Learning_Agent, aca.Actor_Critic_Agent]):
    for G in all_graph:
        print("         ", G)
        for ti, aa in zip(["reverse", "kakusa", "big_reverse", "infration"], all_after):
            for k in range(5):
                RESULT_NAME = RESULT_DIR+G+"/"+alg_name+"/"+str(k)+"/nipd_"+ti
                if not os.path.exists("/".join(RESULT_NAME.split("/"))[:-1]):
                    os.makedirs("/".join(RESULT_NAME.split("/"))[:-1])
                W = synchro_world.synchro_world(100, 1000, [2,-2,2,2], all_graph[G], alg,
                                                share_rate=0.5, altered_func=aa)
                W.run()
                W.save(RESULT_NAME)

print('done!!')
