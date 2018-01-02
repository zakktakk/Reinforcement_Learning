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

"""payoff matrix"""
from world.payoff_matrix import *

graph_prefix = "../src/networks/"

# graphの定義
pcG = network_utils.graph_generator.powerlaw_cluster_graph

all_p = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]


RESULT_DIR = "../results/share_q_noise/powerlaw_cluster/q"
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

for p in all_p:
    for k in range(5):
        RESULT_NAME = RESULT_DIR+"/"+str(k)+"/prisoners_dilemma_"+str(p*100)
        if not os.path.exists("/".join(RESULT_NAME.split("/")[:-1])):
            os.makedirs("/".join(RESULT_NAME.split("/")[:-1]))
        W = synchro_world.synchro_world(100, 1000, prisoners_dilemma(), pcG, ql.Q_Learning_Agent, share_rate=p)
        W.run()
        W.save(RESULT_NAME)

print('done!!')