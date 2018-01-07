#-*- coding: utf-8 -*-
# author : Takuro Yamazaki
# description : シグナリング＋事故

import os
import numpy as np
import pandas as pd
from collections import OrderedDict

"""simulation world"""
from world import synchro_world_signal

"""network"""
from networks import network_utils

"""agent"""
from agent import Q_Learning_Agent as ql
from agent import Actor_Critic_Agent as aca


"""payoff matrix"""
from world.payoff_matrix import *


# graphの定義
pcG = network_utils.graph_generator.powerlaw_cluster_graph

all_after = [[2,2,2,-2], [2,-10,2,10], [10,10,10,-10], [10,-10,10,10]]

# share rateの定義
all_p = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]


RESULT_DIR = "../results/signal_accident/powerlaw_cluster/"

for alg_name, alg in zip(["q", "sarsa"], [ql.Q_Learning_Agent, aca.Actor_Critic_Agent]):
    for ap in all_p:
        for ti, aa in zip(["reverse", "kakusa", "big_reverse", "infration"], all_after):
            for k in range(3):
                RESULT_NAME = RESULT_DIR+alg_name+"/"+str(k)+"/nipd_"+ti+"_"+str(ap*100)
                if not os.path.exists("/".join(RESULT_NAME.split("/"))[:-1]):
                    os.makedirs("/".join(RESULT_NAME.split("/"))[:-1])
                W = synchro_world_signal.synchro_world_signal(100, 5000,[2,-2,2,2] , pcG, alg,
                                                          p_noise=ap, altered_func=aa)
                W.run()
                W.save(RESULT_NAME)

print('done!!')
