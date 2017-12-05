#-*- coding: utf-8 -*-
# author : Takuro Yamazaki
# description : 全員の行動だけが観測可能な場合

import os
from collections import OrderedDict

"""simulation world"""
from world import synchro_world_public

"""network"""
from networks import network_utils

"""agent"""
# defaultはこの4種類にしよう
from agent import Q_Learning_Agent as ql


"""payoff matrix"""
from world.payoff_matrix import *


# graphの定義
rG = network_utils.graph_generator.random_graph

all_after = [pd.DataFrame(np.array([[3, 0],[2, 0]]),index=list('cd'), columns=list('cd')),
             pd.DataFrame(np.array([[30, 10], [5, 1]]), index=list('cd'), columns=list('cd')),
             pd.DataFrame(np.array([[4, 0], [0, 2]]), index=list('cd'), columns=list('cd'))
             ]

# agentの定義
all_agent = {"q":ql.Q_Learning_Agent}

RESULT_DIR = "../results/public_accident/"
for ag in all_agent.keys():
    if not os.path.exists(RESULT_DIR+ag):
        os.makedirs(RESULT_DIR+ag)
    print(ag)

    for ti, aa in zip(["kaishou", "kakudai", "coodinate"], all_after):
        RESULT_NAME = RESULT_DIR+ag+"/prisoner_"+ti
        W = synchro_world_public.synchro_world_public(100, 1000, prisoners_dilemma(), rG, all_agent[ag], altered_mat=aa)
        W.run()
        W.save(RESULT_NAME)

print('done!!')
