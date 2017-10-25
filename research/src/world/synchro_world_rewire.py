#-*- coding: utf-8 -*-
# author : Takuro Yamazaki
# description : 動的ネットワーク上での実験
#
# データに記述すべきメタデータ
# - 繰り返し回数
# - エージェント数
# - ネットワークの種類
# - エージェントの種類
# - エッジ数
# - 利得行列の種類
# - その他の条件(初期行動制約など)


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

"""self made library"""
from .synchro_world import synchro_world


class synchro_world_rewire(synchro_world):
    def __init__(self, n_agent, n_round, payoff_matrix, network_alg, rl_alg):
        super().__init__(n_agent, n_round, payoff_matrix, network_alg, rl_alg)


    def run(self):
        rand = True
        nodes = self.G.nodes()
        for i in range(self.n_round):
            is_rewire = (i % 100 == 0)
            rewire_lst = []

            if i % 1000 == 0: # 途中経過の出力
                print('iter : '+str(i))

            # 全エージェントが同期的に行動選択
            if i == len(list(self.payoff_matrix))*5:  # 初期のランダム行動
                rand = False

            # 全てのエージェントが行動選択
            for n in nodes:
                self.G.node[n]["action"] = self.G.node[n]["agent"].act(0, random=rand)  # for Q Leaning, FAQ
                self.agent_action_table[n][i] = self.G.node[n]["action"]

            # 報酬計算&Q値更新
            for n in nodes:
                neighbors = self.G.neighbors(n)
                n_action = self.G.node[n]["action"]
                n_reward = 0
                n_reward_min = 10000
                n_reward_argmin = 0

                for ne in neighbors:
                    ne_action = self.G.node[ne]["action"]
                    n_ne_payoff = self.payoff_matrix[ne_action][n_action]
                    n_reward += n_ne_payoff
                    if n_reward_min > n_ne_payoff:
                        n_reward_min = n_ne_payoff
                        n_reward_argmin = ne

                rewire_lst.append((n, n_reward_argmin))

                self.G.node[n]["reward"] = n_reward
                self.agent_payoff_table[n][i] = n_reward/len(neighbors)
                self.G.node[n]["agent"].update(0, n_reward) # 今状態は0だけ
                #self.G.node[n]["agent"].update(0, n_reward, n_action) # for SARSA

            # リンクのrewire
            if is_rewire:
                self.neighbors_neighbor(rewire_lst)

    def random_rewire(self, rewire_lst):
        self.G.remove_edges_from(rewire_lst)
        nodes = self.G.nodes()
        for from_, _ in rewire_lst:
            rewire_node = np.random.choice(nodes)
            while(self.G.has_edge(from_, rewire_node)):
                rewire_node = np.random.choice(nodes)
            self.G.add_edge(from_, rewire_node)


    def neighbors_neighbor(self, rewire_lst):
        self.G.remove_edges_from(rewire_lst)
        nodes = self.G.nodes()
        for from_, to_ in rewire_lst:
            neighbors = self.G.neighbors(to_)
            rewire_node = np.random.choice(neighbors)
            while(self.G.has_edge(from_, rewire_node)):
                rewire_node = np.random.choice(neighbors)
            self.G.add_edge(from_, rewire_node)


    def preference_attachment(self, rewire_lst):
        alpha = 1
        nodes = self.G.nodes()
        self.G.remove_edges_from(rewire_lst)
        degree = np.array([self.G.degree(n)+alpha for n in nodes])
        prob = degree / np.sum(degree)

        for from_, to_ in rewire_lst:
            rewire_node = np.random.choice(nodes, p=prob)
            while(rewire_node == from_):
                rewire_node = np.random.choice(nodes, p=prob)
            self.G.add_edge(from_, rewire_node)
