#-*- coding: utf-8 -*-
# author : Takuro Yamazaki
# description : 同時プレイの世界
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


class synchro_world_lr_accident(synchro_world):
    def __init__(self, n_agent, n_round, payoff_matrix, network_alg, rl_alg, after_matrix, gamma):
        self.n_agent = n_agent
        self.n_round = n_round
        self.game_name, self.payoff_matrix = payoff_matrix
        self.network_alg = network_alg
        self.after_matrix = after_matrix
        self.rl_alg = rl_alg
        self.gamma = gamma
        self.create_network()
        self.payoff_table = pd.DataFrame(np.zeros((n_round, 1)))
        self.coop_per_table = pd.DataFrame(np.zeros((n_round, 1)))


    def create_network(self) -> None:
        """
        :description ネットワークモデルの定義
        :param n_agent : エージェントの数
        """
        self.G = self.network_alg()  # change network
        self.n_edges = self.G.size()  # number of edges

        for n in self.G.nodes():
            neighbors = self.G.neighbors(n)
            agent = self.rl_alg(n, np.array([0]), self.payoff_matrix.index, gamma=self.gamma)
            self.G.node[n]["agent"] = agent
            self.G.node[n]["action"] = 0


    def change_payoff_metrix(self):
        self.payoff_matrix = self.after_matrix


    def run(self):
        rand = True
        nodes = self.G.nodes()
        for i in range(self.n_round):
            if i % 1000 == 0: # 途中経過の出力
                print('iter : '+str(i))

            # 全エージェントが同期的に行動選択
            if i == len(list(self.payoff_matrix))*5:  # 初期のランダム行動
                rand = False

            if i == self.n_round * 0.5: # 半分経過したらpayoffを帰る
                self.change_payoff_metrix()

            # 全てのエージェントが行動選択
            coop_num = 0
            for n in nodes:
                self.G.node[n]["action"] = self.G.node[n]["agent"].act(0, random=rand)  # for Q Leaning, FAQ

                if self.G.node[n]["action"] == "c":
                    coop_num += 1

            self.coop_per_table[0][i] = coop_num / self.n_agent

            # 報酬計算&Q値更新
            reward_sum = 0
            for n in nodes:
                neighbors = self.G.neighbors(n)
                n_action = self.G.node[n]["action"]
                n_reward = 0
                for ne in neighbors:
                    ne_action = self.G.node[ne]["action"]
                    n_reward += self.payoff_matrix[ne_action][n_action]

                self.G.node[n]["reward"] = n_reward
                reward_sum += n_reward
                self.G.node[n]["agent"].update(0, n_reward) # 今状態は0だけ
                #self.G.node[n]["agent"].update(0, n_reward, n_action) # for SARSA

            self.payoff_table[0][i] = reward_sum / self.n_agent
