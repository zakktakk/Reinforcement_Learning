#-*- coding: utf-8 -*-
# author : Takuro Yamazaki

import os
from collections import OrderedDict

"""simulation world"""
from world import synchro_world

"""network"""
from networks import network_utils

"""agent"""
from agent import WoLF_PHC_Agent as wpa
from agent import Q_Learning_Agent as ql
from agent import Actor_Critic_Agent as aca
from agent import SARSA_Agent as sarsa

"""payoff matrix"""
from world.payoff_matrix import *

graph_prefix = "../src/networks/"

# graphの定義
pcG = network_utils.graph_generator.powerlaw_cluster_graph

# agentの定義
algs = OrderedDict((("q",ql.Q_Learning_Agent),
                         ("wolf_phc",wpa.WoLF_PHC_Agent),
                         ("actor_critic",aca.Actor_Critic_Agent),
                         ("sarsa", sarsa.SARSA_Agent)))

RESULT_DIR = "../results/alg/powerlaw_cluster/"

for alg in algs.keys():
    for k in range(5):
        RESULT_NAME = RESULT_DIR+alg+"/"+str(k)+"/nipd"

        if not os.path.exists("/".join(RESULT_NAME.split("/")[:-1])):
            os.makedirs("/".join(RESULT_NAME.split("/")[:-1]))

        W = synchro_world.synchro_world(100, 1000, [2,-2,2,2], pcG, algs[alg])
        W.run()
        W.save(RESULT_NAME)

print('done!!')
