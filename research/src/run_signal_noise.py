#-*- coding: utf-8 -*-
# author : Takuro Yamazaki
# description : シグナルの効果

import os
from collections import OrderedDict

"""simulation world"""
from world import synchro_world_signal

"""network"""
from networks import network_utils

"""agent"""
# defaultはこの4種類にしよう
from agent import Actor_Critic_Agent as aca
from agent import Q_Learning_Agent as ql

"""payoff matrix"""
from world.payoff_matrix import *

graph_prefix = "../src/networks/"

# graphの定義
pcG = network_utils.graph_generator.powerlaw_cluster_graph

all_p = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

RESULT_DIR = "../results/signal/powerlaw_cluster/"

if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

for alg_name, alg in zip(["sarsa"], [aca.Actor_Critic_Agent]):
    for p in all_p:
        for k in range(3):
            RESULT_NAME = RESULT_DIR+alg_name+"/"+str(k)+"/nipd_"+str(p*100)

            if not os.path.exists("/".join(RESULT_NAME.split("/")[:-1])):
                os.makedirs("/".join(RESULT_NAME.split("/")[:-1]))

            W = synchro_world_signal.synchro_world_signal(100, 5000, [2,-2,2,2], pcG, alg,
                                                                  p_noise=p)
            W.run()
            W.save(RESULT_NAME)

print('done!!')
