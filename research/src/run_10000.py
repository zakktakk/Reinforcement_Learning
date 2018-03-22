#-*- coding: utf-8 -*-
# author : Takuro Yamazaki
# description : アルゴリズムごとの協調行動の違い

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

# graphの定義
pcG = network_utils.graph_generator.powerlaw_cluster_graph

# accident後の行列
all_after = [[2,2,2,-2], [2,-10,2,10]]


RESULT_DIR = "../results/10000_accident/powerlaw_cluster/q/"

for ti, aa in zip(["reverse", "kakusa"], all_after):
    for k in range(5):
        RESULT_NAME = RESULT_DIR+"/"+str(k)+"/nipd_"+ti

        if not os.path.exists("/".join(RESULT_NAME.split("/")[:-1])):
            os.makedirs("/".join(RESULT_NAME.split("/")[:-1]))

        W = synchro_world.synchro_world(100, 10000, [2,-2,2,2], pcG, ql.Q_Learning_Agent, altered_func=aa)
        W.run()
        W.save(RESULT_NAME)

print('done!!')
