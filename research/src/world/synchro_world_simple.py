#-*- coding: utf-8 -*-
# author : Takuro Yamazaki
# description : 定義通りのゲーム
# ref : file:///Users/yamazakitakurou/Downloads/8708-38013-1-PB.pdf


"""path setting"""
import sys
sys.path.append("../")

"""util libraries"""
import numpy as np
import pandas as pd
import networkx as nx

"""visualize"""
#ディスプレイ表示させない
import matplotlib as mpl
mpl.use('tkagg')

import matplotlib.pyplot as plt
plt.style.use('ggplot')

class synchro_world_simple(object):
    def __init__(self, n_agent, n_round, payoff_matrix, network_alg, rl_alg):
        self.n_agent = n_agent
        self.n_round = n_round
        self.game_name, self.payoff_matrix = payoff_matrix
        self.network_alg = network_alg
        self.rl_alg = rl_alg
        self.create_network()
        self.payoff_table = pd.DataFrame(np.zeros((1, n_agent)))
        self.coop_per_table = pd.DataFrame(np.zeros((1, n_agent)))


    def create_network(self) -> None:
        """
        :description ネットワークモデルの定義
        :param n_agent : エージェントの数
        """
        self.G = self.network_alg()  # change network
        self.n_edges = self.G.size() # number of edges

        for n in self.G.nodes():
            neighbors = self.G.neighbors(n)
            agent = self.rl_alg(n, np.array([0]), self.payoff_matrix.index)
            self.G.node[n]["agent"] = agent
            self.G.node[n]["action"] = 0



    def run(self):
        rand = True
        nodes = self.G.nodes()
        for i in range(self.n_round):
            if i % 1000 == 0: # 途中経過の出力
                print('iter : '+str(i))

            # 全エージェントが同期的に行動選択
            if i == len(list(self.payoff_matrix))*5:  # 初期のランダム行動
                rand = False

            # 全てのエージェントが隣接エージェントと対戦＆更新
            for n1 in nodes:
                n2 = np.random.choice(self.G.neighbors(n1))
                
                n1_action = self.G.node[n1]["agent"].act(0, random=rand)
                n2_action = self.G.node[n2]["agent"].act(0, random=rand)

                n1_reward = self.payoff_matrix[n2_action][n1_action]
                n2_reward = self.payoff_matrix[n1_action][n2_action]

                self.G.node[n1]["agent"].update(0, n1_reward) # 今状態は0だけ
                #self.G.node[n1]["agent"].update(0, n1_reward, n1_action) # for SARSA

                self.G.node[n2]["agent"].update(0, n2_reward)
                #self.G.node[n2]["agent"].update(0, n2_reward, n2_action) # for SARSA

