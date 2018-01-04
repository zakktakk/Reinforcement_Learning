#-*- coding: utf-8 -*-
# author : Takuro Yamazaki
# description : Q値を共有した時、事故が起きるシナリオ

import os
import numpy as np
import pandas as pd
from collections import OrderedDict

"""simulation world"""
from world import synchro_world

"""network"""
from networks import network_utils

"""agent"""
from agent import Q_Learning_Agent as ql

"""payoff matrix"""
from world.payoff_matrix import *


# graphの定義
pcG = network_utils.graph_generator.powerlaw_cluster_graph

all_after = [pd.DataFrame(np.array([[1, 5],[0, 3]]),index=list('cd'), columns=list('cd')),
             pd.DataFrame(np.array([[6, 3], [5, 1]]), index=list('cd'), columns=list('cd')),
             pd.DataFrame(np.array([[15, 0], [25, 5]]), index=list('cd'), columns=list('cd')),
             pd.DataFrame(np.array([[3, 2], [4, 0]]), index=list('cd'), columns=list('cd'))
             ]


# share rateの定義
all_share_rate = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]


RESULT_DIR = "../results/share_q_accident/powerlaw_cluster/q"
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

for asr in all_share_rate:
    print("         ", asr)
    for ti, aa in zip(["reverse", "biggerc", "infration", "chicken"], all_after):
        for k in range(5):
            RESULT_NAME = RESULT_DIR+"/"+str(k)+"/prisoner"+"_"+str(asr*100)+"_"+ti
            if not os.path.exists("/".join(RESULT_NAME.split("/"))[:-1]):
                os.makedirs("/".join(RESULT_NAME.split("/"))[:-1])
            W = synchro_world.synchro_world(100, 1000, prisoners_dilemma(), pcG, ql.Q_Learning_Agent,
                                            share_rate=asr, altered_mat=aa)
            W.run()
            W.save(RESULT_NAME)

print('done!!')
