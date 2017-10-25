#-*- coding: utf-8 -*-
# author : Takuro Yamazaki
# description : 基礎的な実験

import os
from collections import OrderedDict

"""simulation world"""
from world import synchro_world

"""network"""
from networks import network_utils

"""agent"""
# defaultはこの4種類にしよう
from agent import Actor_Critic_Agent as aca
from agent import WoLF_PHC_Agent as wpa
from agent import Q_Learning_Agent as ql
from agent import SARSA_Agent as sarsa

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
all_agent = OrderedDict((("q",ql.Q_Learning_Agent),("actor_critic",aca.Actor_Critic_Agent), ("wplf_phc",wpa.WoLF_PHC_Agent)))
# all_agent = {"sarsa":sarsa.SARSA_Agent}

RESULT_DIR = "../results/basic/"
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
            RESULT_NAME = RESULT_DIR+ag+"/"+G+"/"+g
            W = synchro_world.synchro_world(100, 5000, eval(g)(), all_graph[G], all_agent[ag])
            W.run()
            W.save(RESULT_NAME)

print('done!!')
