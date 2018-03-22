#-*- coding: utf-8 -*-
# author : Takuro Yamazaki

import os
from collections import OrderedDict

"""simulation world"""
from world import synchro_world

"""network"""
from networks import network_utils

"""agent"""
from agent import Actor_Critic_Agent_TD as acatd
from agent import SARSA_Agent_TD as sarsatd

"""payoff matrix"""
from world.payoff_matrix import *


# graphの定義
pcG = network_utils.graph_generator.powerlaw_cluster_graph

all_alpha = [0.1, 0.3, 0.5, 0.7, 0.9]

# accident後の行列
all_after = [[2,2,2,-2], [2,-10,2,10], [10,10,10,-10], [10,-10,10,10]]


RESULT_DIR = "../results/lmd_accident/powerlaw_cluster/"


for alg_name, alg in zip(["actor_critic"], [acatd.Actor_Critic_Agent_TD]):
    for ti, aa in zip(["reverse", "kakusa", "big_reverse", "infration"], all_after):
        for l in all_alpha:
            print("      ", l)

            for k in range(5):
                RESULT_NAME = RESULT_DIR+alg_name+"/"+str(k)+"/nipd_"+ti+"_"+str(l*100)

                if not os.path.exists("/".join(RESULT_NAME.split("/")[:-1])):
                    os.makedirs("/".join(RESULT_NAME.split("/")[:-1]))
                
                W = synchro_world.synchro_world(100, 1000, [2,-2, 2, 2], pcG, alg, rl_param=dict(lmd=l), altered_func=aa)
                W.run()
                W.save(RESULT_NAME)

print('done!!')
