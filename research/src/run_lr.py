#-*- coding: utf-8 -*-
# author : Takuro Yamazaki
# description : 学習率を変化させた時の学習の差異

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


# graphの定義
rG = network_utils.graph_generator.random_graph

# payoffmatrixの定義
all_matrix = ["prisoners_dilemma", "coodination_game", "PG", "CG"]

# agentの定義
all_agent = {"q":ql.Q_Learning_Agent}

# 割引率gammaの定義
all_gamma = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98]


RESULT_DIR = "../results_1123/lr/random/"
for ag in all_agent.keys():
    if not os.path.exists(RESULT_DIR+ag):
        os.makedirs(RESULT_DIR+ag)
    print(ag)
    for g in all_matrix:
        print("    "+g)
        for gamma in all_gamma:
            print("      ", gamma)
            RESULT_NAME = RESULT_DIR+ag+"/"+g+"_"+str(gamma*100)
            W = synchro_world.synchro_world(100, 1000, eval(g)(), rG, all_agent[ag], rl_param=dict(gamma=gamma))
            W.run()
            W.save(RESULT_NAME)

print('done!!')
