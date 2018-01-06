#-*- coding: utf-8 -*-
# author : Takuro Yamazaki
# description : 事故が起こった場合、アルゴリズムごとの学習の違い

import os
import itertools
import pandas as pd
import numpy as np
from collections import OrderedDict

"""simulation world"""
from world import synchro_world

"""network"""
from networks import network_utils

"""agent"""
# defaultはこの4種類にしよう
from agent import WoLF_PHC_Agent as wpa
from agent import Q_Learning_Agent as ql
from agent import Actor_Critic_Agent as aca
from agent import SARSA_Agent as sarsa

"""payoff matrix"""
from world.payoff_matrix import *


graph_prefix = "../src/networks/"

# graphの定義
pcG = network_utils.graph_generator.powerlaw_cluster_graph


# accident後の行列
all_after = [[3,3,2,2]]

# agentの定義
all_agent = OrderedDict((("q",ql.Q_Learning_Agent),
                         ("wolf_phc",wpa.WoLF_PHC_Agent),
                         ("actor_critic",aca.Actor_Critic_Agent),
                         ("sarsa", sarsa.SARSA_Agent)))


RESULT_DIR = "../results/accident_alg/powerlaw_cluster/"
for ag in all_agent.keys():
    for ti, aa in zip(["reverse", "biggerc", "infration", "chicken"], all_after):

        for k in range(5):
            RESULT_NAME = RESULT_DIR+ag+"/"+str(k)+"/prisoners_dilemma_"+ti

            if not os.path.exists("/".join(RESULT_NAME.split("/")[:-1])):
                os.makedirs("/".join(RESULT_NAME.split("/")[:-1]))

            W = synchro_world.synchro_world(100, 1000, [2,-2,2,2], pcG, all_agent[ag], altered_mat=aa)
            W.run()
            W.save(RESULT_NAME)

print('done!!')
