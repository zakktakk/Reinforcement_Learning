#-*- coding: utf-8 -*-
# author : Takuro Yamazaki
# description : ランダムグラフのときノード数に変化によりどう影響が出るか

import os
from collections import OrderedDict

"""simulation world"""
from world import synchro_world

"""network"""
from networks import network_utils

"""agent"""
# defaultはこの4種類にしよう
from agent import WoLF_PHC_Agent as wpa
from agent import Q_Learning_Agent as ql
from agent import SARSA_Agent as sarsa

"""payoff matrix"""
from world.payoff_matrix import *


# graphの定義
pcG = network_utils.graph_generator.powerlaw_cluster_graph


# payoffmatrixの定義
all_matrix = ["prisoners_dilemma"]


# agentの定義
all_agent = OrderedDict((("q",ql.Q_Learning_Agent)))


# エージェント数の定義
all_agent_num = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200]


RESULT_DIR = "../results/agent_num/powerlaw_cluster/"
for ag in all_agent.keys():
    if not os.path.exists(RESULT_DIR+ag):
        os.makedirs(RESULT_DIR+ag)

    for g in all_matrix:
        print("    "+g)
        for a_num in all_agent_num:
            print("     ", str(a_num))
            RESULT_NAME = RESULT_DIR+ag+"/"+g+"_"+str(a_num)
            W = synchro_world.synchro_world(a_num, 1000, eval(g)(), pcG, all_agent[ag], nwk_param=dict(n=a_num, m=a_num*30))
            W.run()
            W.save(RESULT_NAME)

print('done!!')
