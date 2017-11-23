#-*- coding: utf-8 -*-
# author : Takuro Yamazaki
# description : 実験2、グラフ構造とアルゴリズムごとの結果の違い

import os
import numpy as np
import pandas as pd
from collections import OrderedDict

"""simulation world"""
from world import synchro_world_observable

"""network"""
from networks import network_utils

"""agent"""
# defaultはこの4種類にしよう
from agent import WoLF_PHC_Agent as wpa
from agent import Q_Learning_Agent as ql
from agent import SARSA_Agent as sarsa

"""payoff matrix"""
from world.payoff_matrix import *


# graphの定義
rG = network_utils.graph_generator.random_graph

all_after = [pd.DataFrame(np.array([[3, 0],[5, 0]]),index=list('cd'), columns=list('cd'))]

all_graph = {"random":rG}

RESULT_NAME = "../results/observable_accident/random/q/prisoners_kaishou"

W = synchro_world_observable.synchro_world_observable(100, 1000, prisoners_dilemma(), rG,
                                                      ql.Q_Learning_Agent, altered_mat=all_after)
W.run()
W.save(RESULT_NAME)

print('done!!')
