#-*- coding: utf-8 -*-
# author : Takuro Yamazaki
# last update : 06/15/2017
# description : 一つ前の行動を知らせる(関係してるだけで意味を知っているわけではない), 数だけ知覚できる
# TODO coop側の人しかsignal出せないかつ，signal出す出さないは選択可能，signalはコストがかかる -> バランス取らないと初期の行動で行動割合の偏りが生じるぞ
# TODO 全体の助教が見える場合、近傍の状況が見える場合
# 分離させてシグナルを発信する人にランダムにリンクを張り替えるを加える？
# :thinking: signalとactionは別で学習したほうが良い？

#データに記述すべきメタデータ
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


class synchro_world_preaction(synchro_world):
    def __init__(self, n_agent, n_round, payoff_matrix, network_alg, rl_alg, p_noise=-1):
        super().__init__(n_agent, n_round, payoff_matrix, network_alg, rl_alg)
        self.__p_noise = p_noise


    def create_network(self):
        """
        :description ネットワークモデルの定義
        :param n_agent : エージェントの数
        """
        self.G = self.network_alg()  # change network
        self.n_edges = self.G.size()  # number of edges

        for n in self.G.nodes():
            neighbors = self.G.neighbors(n)
            # 0~隣接エージェント数が状態，シグナルが届いた数を表す
            agent = self.rl_alg(n, np.arange(len(neighbors)+1), self.payoff_matrix.index)
            self.G.node[n]["agent"] = agent
            self.G.node[n]["action"] = 0
            self.G.node[n]["n_signal"] = 0 #状態の初期値は0


    def run(self):
        rand = True
        nodes = self.G.nodes()

        for i in range(self.n_round):
            if i % 1000 == 0:
                print('iter : '+str(i))

            #全エージェントが同期的に行動選択
            if i == len(self.payoff_matrix)*5: #初期のランダム行動
                rand=False

            for n in  nodes:
                # TODO n_signalは重み付きにした方がいいのか？
                self.G.node[n]["action"] = self.G.node[n]["agent"].act(self.G.node[n]["n_signal"], random=rand) #reduction=Trueで減衰
                self.agent_action_table[n][i] = self.G.node[n]["action"]

            #報酬計算&Q値更新
            for n in nodes:
                neighbors = self.G.neighbors(n)
                n_action = self.G.node[n]["action"]
                n_reward = 0
                n_signal = 0
                for ne in neighbors:
                    ne_action = self.G.node[ne]["action"]

                    # noisyなsignal
                    noise = np.random.rand()
                    is_noise = (self.__p_noise > noise) #Trueだったらnoise
                    if ne_action == "c":
                        if not is_noise:
                            n_signal += 1
                    else:
                        if is_noise:
                            n_signal += 1

                    n_reward += self.payoff_matrix[n_action][ne_action]

                self.G.node[n]["reward"] = n_reward
                self.agent_payoff_table[n][i] = n_reward/len(neighbors)
                self.G.node[n]["n_signal"] = n_signal

                self.G.node[n]["agent"].update(self.G.node[n]["n_signal"], n_reward)
                #self.G.node[n]["agent"].update(self.G.node[n]["n_signal"], n_reward, n_action) # for SARSA
