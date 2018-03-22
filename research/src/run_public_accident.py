#-*- coding: utf-8 -*-
# author : Takuro Yamazaki
# description : 全員の行動だけが観測可能な場合

import os
import numpy as np
import pandas as pd

"""simulation world"""
from world import synchro_world_public

"""network"""
from networks import network_utils

"""agent"""
# defaultはこの4種類にしよう
from agent import Q_Learning_Agent as ql

from collections import OrderedDict

"""payoff matrix"""
from world.payoff_matrix import *

graph_prefix = "../src/networks/"

# graphの定義
cG = network_utils.graph_generator.complete_graph
pcG = network_utils.graph_generator.powerlaw_cluster_graph

all_p = [0.6]


all_after = [[2,2,2,-2], [2,-10,2,10]]


RESULT_DIR = "../results/share_q_public_accident/powerlaw/q/"

for p in all_p:
    for ti, aa in zip(["reverse", "kakusa"], all_after):
        for k in range(5):
            RESULT_NAME = RESULT_DIR+str(k)+"/nipd_"+ti+"_"+str(100*p)
            if not os.path.exists("/".join(RESULT_NAME.split("/")[:-1])):
                os.makedirs("/".join(RESULT_NAME.split("/")[:-1]))
            W = synchro_world_public.synchro_world_public(100, 1000, [2,-2,2,2], pcG,ql.Q_Learning_Agent,
                                                          altered_func=aa, share_rate=p)
            W.run()
            W.save(RESULT_NAME)

print('done!!')
