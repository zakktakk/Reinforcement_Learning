#-*- coding: utf-8 -*-
# author : Takuro Yamazaki
# description : 学習率を変化させた時の学習の差異

import os
import numpy as np
import pandas as pd

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

# agentの定義
all_agent = {"q":ql.Q_Learning_Agent}

all_after = [pd.DataFrame(np.array([[3, 0],[5, 0]]),index=list('cd'), columns=list('cd'))]

# 割引率gammaの定義
all_gamma = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98]


RESULT_DIR = "../results/lr_accident/random/"
for ag in all_agent.keys():
    if not os.path.exists(RESULT_DIR+ag):
        os.makedirs(RESULT_DIR+ag)
    print(ag)

    for gamma in all_gamma:
        for ti, aa in zip(["kaishou"], all_after):
            RESULT_NAME = RESULT_DIR+ag+"/prisoner_"+str(gamma*100)+"_"+ti
            W = synchro_world.synchro_world(100, 1000, prisoners_dilemma(), rG, all_agent[ag],
                                                                    altered_mat=aa,rl_param=dict(gamma=gamma))
            W.run()
            W.save(RESULT_NAME)

print('done!!')
