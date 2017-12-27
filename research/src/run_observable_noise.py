#-*- coding: utf-8 -*-
# author : Takuro Yamazaki
# description : 実験2、グラフ構造とアルゴリズムごとの結果の違い

import os
from collections import OrderedDict

"""simulation world"""
from world import synchro_world_observable

"""network"""
from networks import network_utils

"""agent"""
# defaultはこの4種類にしよう
from agent import WoLF_PHC_Agent as wpa
from agent import Q_Learning_Agent as ql
from agent import Actor_Critic_Agent as aca
from agent import SARSA_Agent as sarsa

"""payoff matrix"""
from world.payoff_matrix import *

graph_prefix = "../src/networks/"

# graphの定義
pcG = network_utils.graph_generator.powerlaw_cluster_graph

all_p = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


# agentの定義
all_agent = {"q": ql.Q_Learning_Agent}

RESULT_DIR = "../results/observable_noise/powerlaw_cluster"
for ag in all_agent.keys():
    if not os.path.exists(RESULT_DIR+ag):
        os.makedirs(RESULT_DIR+ag)
    print(ag)
    for p in all_p:
        RESULT_NAME = RESULT_DIR+ag+"/prisoners_dilemma_"+str(p*100)
        W = synchro_world_observable.synchro_world_observable(100, 1000, prisoners_dilemma(), pcG, all_agent[ag])
        W.run()
        W.save(RESULT_NAME)

print('done!!')
