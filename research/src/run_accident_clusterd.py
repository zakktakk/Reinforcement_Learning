#-*- coding: utf-8 -*-
# author : Takuro Yamazaki
# description : 基礎的な実験

import os
import itertools
import pandas as pd
import numpy as np
from collections import OrderedDict

"""simulation world"""
from world import synchro_world_accident_clusterd

"""network"""
from networks import network_utils

"""agent"""
# defaultはこの4種類にしよう
from agent import Q_Learning_Agent as ql

"""payoff matrix"""
from world.payoff_matrix import *


prefix = "../src/networks/"
# graphの定義
all_graph = ["inner_dence_clustered.gpickle", "multiple_clustered.gpickle"]

all_after = [pd.DataFrame(np.array([[3, 0],[5, 0]]),index=list('cd'), columns=list('cd'))]

# agentの定義
all_agent = {"q":ql.Q_Learning_Agent}

RESULT_DIR = "../results/accident_clustered/"
for ag in all_agent.keys():
    if not os.path.exists(RESULT_DIR+ag):
        os.makedirs(RESULT_DIR+ag)
    print(ag)
    for G in all_graph:
        G_name = G.split(".")[0]
        if not os.path.exists(RESULT_DIR + ag + "/" + G_name):
            os.makedirs(RESULT_DIR + ag + "/" + G_name)
        print("  "+G_name)
        for ti, aa in zip(["kaishou"], all_after):
            RESULT_NAME = RESULT_DIR+ag+"/"+G_name+"/prisoners_dilemma_"+ti
            W = synchro_world_accident_clusterd.synchro_world_accident_clustered(100, 2000, prisoners_dilemma(), prefix+G, all_agent[ag], aa)
            W.run()
            W.save(RESULT_NAME)

print('done!!')
