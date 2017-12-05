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
# defaultはこの4種類にしよう
from agent import WoLF_PHC_Agent as wpa
from agent import Q_Learning_Agent as ql
from agent import SARSA_Agent as sarsa

"""payoff matrix"""
from world.payoff_matrix import *


# graphの定義
rG = network_utils.graph_generator.random_graph

# payoffmatrixの定義
all_matrix = ["prisoners_dilemma", "coodination_game"]

# agentの定義
all_agent = OrderedDict((("q",ql.Q_Learning_Agent), ("wolf_phc",wpa.WoLF_PHC_Agent)))
# all_agent = {"sarsa":sarsa.SARSA_Agent}

# エージェント数の定義
all_edge_num = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500]


RESULT_DIR = "../results/edge_num/random/"
for ag in all_agent.keys():
    if not os.path.exists(RESULT_DIR+ag):
        os.makedirs(RESULT_DIR+ag)

    for g in all_matrix:
        print("    "+g)
        for e_num in all_edge_num:
            print("     ", str(e_num))
            RESULT_NAME = RESULT_DIR+ag+"/"+g+"_"+str(e_num)
            W = synchro_world.synchro_world(100, 1000, eval(g)(), rG, all_agent[ag], nwk_param=dict(m=e_num))
            W.run()
            W.save(RESULT_NAME)

print('done!!')
