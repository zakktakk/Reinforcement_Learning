#-*- coding: utf-8 -*-
# author : Takuro Yamazaki
# description : ランダムグラフのときエッジ数によりどのような違いが出るのか

import os
from collections import OrderedDict

"""simulation world"""
from world import synchro_world

"""network"""
from networks import network_utils

"""agent"""
from agent import Actor_Critic_Agent as aca
from agent import Q_Learning_Agent as ql

"""payoff matrix"""
from world.payoff_matrix import *


# graphの定義
rG = network_utils.graph_generator.random_graph


# エージェント数の定義
all_edge_num = [500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]


RESULT_DIR = "../results/edge_num/random/q/"

for alg_name, alg in zip(["q", "sarsa"], [ql.Q_Learning_Agent, aca.Actor_Critic_Agent]):
    for e_num in all_edge_num:
        print("     ", str(e_num))

        for k in range(5):
            RESULT_NAME = RESULT_DIR+alg_name+"/"+str(k)+"/nipd_"+str(e_num)
            if not os.path.exists("/".join(RESULT_NAME.split("/")[:-1])):
                os.makedirs("/".join(RESULT_NAME.split("/")[:-1]))
            W = synchro_world.synchro_world(100, 1000, [2,-2,2,2], rG, alg, nwk_param=dict(m=e_num))
            W.run()
            W.save(RESULT_NAME)

print('done!!')
