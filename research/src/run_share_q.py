#-*- coding: utf-8 -*-
# author : Takuro Yamazaki
# description : グラフ構造とアルゴリズムごとの結果の違い

import os
from collections import OrderedDict

"""simulation world"""
from world import synchro_world_share_q

"""network"""
from networks import network_utils

"""agent"""
from agent import Q_Learning_Agent as ql

"""payoff matrix"""
from world.payoff_matrix import *


# graphの定義
cG = network_utils.graph_generator.complete_graph
rG = network_utils.graph_generator.random_graph
g2G = network_utils.graph_generator.grid_2d_graph
pcG = network_utils.graph_generator.powerlaw_cluster_graph

all_graph = OrderedDict((("random",rG), ("grid2d",g2G), ("powerlaw_cluster",pcG), ("complete",cG)))

# payoffmatrixの定義
all_matrix = ["prisoners_dilemma", "coodination_game"]

# agentの定義
all_agent = {"q":ql.Q_Learning_Agent}

# share rateの定義
all_share_rate = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


RESULT_DIR = "../results/share_q/"
for ag in all_agent.keys():
    if not os.path.exists(RESULT_DIR+ag):
        os.makedirs(RESULT_DIR+ag)
    print(ag)
    for G in all_graph:
        if not os.path.exists(RESULT_DIR+ag+"/"+G):
            os.makedirs(RESULT_DIR+ag+"/"+G)
        print("  "+G)
        for g in all_matrix:
            print("    "+g)
            for asr in all_share_rate:
                print("         ", asr)
                RESULT_NAME = RESULT_DIR+ag+"/"+G+"/"+g+"_"+str(asr*100)
                W = synchro_world_share_q.synchro_world_share_q(100, 2000, eval(g)(), all_graph[G], all_agent[ag], asr)
                W.run()
                W.save(RESULT_NAME)

print('done!!')
