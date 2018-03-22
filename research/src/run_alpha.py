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
from agent import Actor_Critic_Agent as aca
from agent import Q_Learning_Agent as ql


"""payoff matrix"""
from world.payoff_matrix import *


# graphの定義
pcG = network_utils.graph_generator.powerlaw_cluster_graph

all_alpha = [0.1, 0.3, 0.5, 0.7, 0.9]


RESULT_DIR = "../results/alpha/powerlaw_cluster/"

for alg_name, alg in zip(["q", "actor"], [ql.Q_Learning_Agent,aca.Actor_Critic_Agent]):
    print(alg_name)
    for l in all_alpha:
        print("      ", l)

        for k in range(5):
            RESULT_NAME = RESULT_DIR+alg_name+"/"+str(k)+"/nipd_"+str(l*100)

            if not os.path.exists("/".join(RESULT_NAME.split("/")[:-1])):
                os.makedirs("/".join(RESULT_NAME.split("/")[:-1]))
            
            W = synchro_world.synchro_world(100, 1000, [2,-2, 2, 2], pcG, alg, rl_param=dict(alpha=l))
            W.run()
            W.save(RESULT_NAME)

print('done!!')
