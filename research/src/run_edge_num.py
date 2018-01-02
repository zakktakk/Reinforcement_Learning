#-*- coding: utf-8 -*-
# author : Takuro Yamazaki
# description : ランダムグラフのときエッジ数によりどのような違いが出るのか

import os
from collections import OrderedDict

"""simulation world"""
from world import synchro_world

"""network"""
from networks import network_utils

"""agent"""
from agent import Q_Learning_Agent as ql

"""payoff matrix"""
from world.payoff_matrix import *


# graphの定義
rG = network_utils.graph_generator.random_graph


# エージェント数の定義
all_edge_num = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500]


RESULT_DIR = "../results/edge_num/random/q/"
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

for e_num in all_edge_num:
    print("     ", str(e_num))
    for k in range(5):
        RESULT_NAME = RESULT_DIR+"/"+str(k)+"/prisoners_dilemma_"+str(e_num)
        if not os.path.exists("/".join(RESULT_NAME.split("/")[:-1])):
            os.makedirs("/".join(RESULT_NAME.split("/")[:-1]))
        W = synchro_world.synchro_world(100, 1000, prisoners_dilemma(), rG, ql.Q_Learning_Agent, nwk_param=dict(m=e_num))
        W.run()
        W.save(RESULT_NAME)

print('done!!')
