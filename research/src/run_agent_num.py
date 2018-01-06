#-*- coding: utf-8 -*-
# author : Takuro Yamazaki
# description : ランダムグラフのときノード数に変化によりどう影響が出るか

import os
from collections import OrderedDict

"""simulation world"""
from world import synchro_world

"""network"""
from networks import network_utils

"""agent"""
from agent import Actor_Critic_Agent as aca
from agent import Q_Learning_Agent as ql


# graphの定義
rG = network_utils.graph_generator.random_graph


# エージェント数の定義
agent_nums = [50, 60, 70, 80, 90, 100, 150, 200]


RESULT_DIR = "../results/agent_num/random/"

for alg_name, alg in zip(["q", "sarsa"], [ql.Q_Learning_Agent, aca.Actor_Critic_Agent]):
    for a in agent_nums:
        print(str(a))

        for k in range(5):
            RESULT_NAME = RESULT_DIR+alg_name+"/"+str(k)+"/nipd_"+str(a)

            if not os.path.exists("/".join(RESULT_NAME.split("/")[:-1])):
                os.makedirs("/".join(RESULT_NAME.split("/")[:-1]))

            W = synchro_world.synchro_world(a, 1000, [2,-2,2,2], rG, alg, nwk_param=dict(n=a, m=a*30))
            W.run()
            W.save(RESULT_NAME)

print('done!!')
