#-*- coding: utf-8 -*-
# author : Takuro Yamazaki
# description : グラフ構造とアルゴリズムごとの結果の違い(cluster graph)

import os
from collections import OrderedDict

"""simulation world"""
from world import synchro_world_clustered

"""agent"""
# defaultはこの4種類にしよう
from agent import Actor_Critic_Agent as aca
from agent import WoLF_PHC_Agent as wpa
from agent import Q_Learning_Agent as ql
from agent import SARSA_Agent as sarsa

"""payoff matrix"""
from world.payoff_matrix import *

prefix = "../src/networks/"
# graphの定義
all_graph = ["inner_dence_clustered.gpickle", "multiple_clustered.gpickle"]

# payoffmatrixの定義
all_matrix = ["prisoners_dilemma", "coodination_game"]

# agentの定義
all_agent = OrderedDict((("q",ql.Q_Learning_Agent),("actor_critic",aca.Actor_Critic_Agent), ("wplf_phc",wpa.WoLF_PHC_Agent)))
# all_agent = {"sarsa":sarsa.SARSA_Agent}

RESULT_DIR = "../results/graph/"
for ag in all_agent.keys():
    if not os.path.exists(RESULT_DIR+ag):
        os.makedirs(RESULT_DIR+ag)
    print(ag)
    for G in all_graph:
        G_name = G.split(".")[0]
        if not os.path.exists(RESULT_DIR+ag+"/"+G_name):
            os.makedirs(RESULT_DIR+ag+"/"+G_name)
        print("  "+G_name)
        for g in all_matrix:
            print("    "+g)
            RESULT_NAME = RESULT_DIR+ag+"/"+G_name+"/"+g
            W = synchro_world_clustered.synchro_world_clustered(100, 5000, eval(g)(), prefix+G, all_agent[ag])
            W.run()
            W.save(RESULT_NAME)

print('done!!')
