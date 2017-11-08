#-*- coding: utf-8 -*-
# author : Takuro Yamazaki
# description : 基礎的な実験

import os
import itertools
import pandas as pd
import numpy as np
from collections import OrderedDict

"""simulation world"""
from world import synchro_world_accident

"""network"""
from networks import network_utils

"""agent"""
# defaultはこの4種類にしよう
from agent import Q_Learning_Agent as ql

"""payoff matrix"""
from world.payoff_matrix import *


# graphの定義
cG = network_utils.graph_generator.complete_graph
rG = network_utils.graph_generator.random_graph
g2G = network_utils.graph_generator.grid_2d_graph
pcG = network_utils.graph_generator.powerlaw_cluster_graph

all_graph = OrderedDict((("random",rG), ("grid2d",g2G), ("powerlaw_cluster",pcG), ("complete",cG)))

all_after = [pd.DataFrame(np.array([[1, 5],[0, 3]]),index=list('cd'), columns=list('cd')),
             pd.DataFrame(np.array([[3, 0], [5, 4]]), index=list('cd'), columns=list('cd'))]

# agentの定義
all_agent = {"q":ql.Q_Learning_Agent}

RESULT_DIR = "../results/accident/"
for ag in all_agent.keys():
    if not os.path.exists(RESULT_DIR+ag):
        os.makedirs(RESULT_DIR+ag)
    print(ag)
    for G in all_graph:
        if not os.path.exists(RESULT_DIR+ag+"/"+G):
            os.makedirs(RESULT_DIR+ag+"/"+G)
        print("  "+G)
        for ti, aa in zip(["tenchi", "kaishou"], all_after):
            RESULT_NAME = RESULT_DIR+ag+"/"+G+"/prisoners_dilemma_"+ti
            W = synchro_world_accident.synchro_world_accident(100, 5000, prisoners_dilemma(), all_graph[G], all_agent[ag], aa)
            W.run()
            W.save(RESULT_NAME)

print('done!!')
