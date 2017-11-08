#-*- coding: utf-8 -*-
# author : Takuro Yamazaki
# description : ランダムグラフのときノード数に変化によりどう影響が出るか

import os
from collections import OrderedDict

"""simulation world"""
from world import synchro_world_edge_num

"""network"""
from networks import network_utils

"""agent"""
# defaultはこの4種類にしよう
from agent import Q_Learning_Agent as ql

"""payoff matrix"""
from world.payoff_matrix import *


# graphの定義
rG = network_utils.graph_generator.random_graph

# payoffmatrixの定義
all_matrix = ["prisoners_dilemma", "coodination_game"]

# agentの定義
all_agent = {"q":ql.Q_Learning_Agent}

# エージェント数の定義
all_agent_num = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200]

RESULT_DIR = "../results/agent_num/random/"
for ag in all_agent.keys():
    if not os.path.exists(RESULT_DIR+ag):
        os.makedirs(RESULT_DIR+ag)

    for g in all_matrix:
        print("    "+g)
        for a_num in all_agent_num:
            print("     ", str(a_num))
            RESULT_NAME = RESULT_DIR+ag+"/"+g+"_"+str(a_num)
            W = synchro_world_edge_num.synchro_world_edge_num(a_num, 5000, a_num*25, eval(g)(), rG, all_agent[ag])
            W.run()
            W.save(RESULT_NAME)

print('done!!')
