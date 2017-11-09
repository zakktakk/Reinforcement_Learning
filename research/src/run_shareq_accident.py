#-*- coding: utf-8 -*-
# author : Takuro Yamazaki
# description : グラフ構造とアルゴリズムごとの結果の違い

import os
import numpy as np
import pandas as pd
from collections import OrderedDict

"""simulation world"""
from world import synchro_world_shareq_accident

"""network"""
from networks import network_utils

"""agent"""
from agent import Q_Learning_Agent as ql

"""payoff matrix"""
from world.payoff_matrix import *


# graphの定義
rG = network_utils.graph_generator.random_graph

all_graph = {"random":rG}


all_after = [pd.DataFrame(np.array([[3, 0],[5, 0]]),index=list('cd'), columns=list('cd'))]

# payoffmatrixの定義
all_matrix = ["prisoners_dilemma"]

# agentの定義
all_agent = {"q":ql.Q_Learning_Agent}

# share rateの定義
all_share_rate = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


RESULT_DIR = "../results/share_q_accident/"
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
                for ti, aa in zip(["kaishou"], all_after):
                    RESULT_NAME = RESULT_DIR+ag+"/"+G+"/"+g+"_"+str(asr*100)+"_"+ti
                    W = synchro_world_shareq_accident.synchro_world_shareq_accident(100, 2000, eval(g)(), all_graph[G], all_agent[ag], asr, aa)
                    W.run()
                    W.save(RESULT_NAME)

print('done!!')
