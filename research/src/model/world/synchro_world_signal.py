# -*- coding: utf-8 -*-
# author : Takuro Yamazaki
# description : noise条件下での動作

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
from synchro_world import synchro_world
from payoff_matrix import *
# from agent.FAQ_Agent import FAQ_Agent
# from agent.SARSA_Agent import SARSA_Agent
# from agent.WoLF_PHC_Agent import WoLF_PHC_Agent
from agent.Q_Learning_Agent import Q_Learning_Agent


class synchro_world_signal(synchro_world):
    def __init__(self, n_agent, n_round, payoff_matrix, network_alg, rl_alg, p_noise=-1):
        super().__init__(n_agent, n_round, payoff_matrix, network_alg, rl_alg)
        self.__p_noise = p_noise


    def __create_network(self) -> None:
        """
        :description ネットワークモデルの定義
        :param n_agent : エージェントの数
        """
        self.G = self.__network_alg(self.__n_agent, 70)  # change network
        self.n_edges = self.G.size()  # number of edges

        for n in self.G.nodes():
            neighbors = self.G.neighbors(n)
            # 0~隣接エージェント数が状態，シグナルが届いた数を表す
            agent = self.__rl_alg(n, np.arange(len(neighbors)+1), self.__payoff_matrix.index)
            self.G.node[n]["agent"] = agent
            self.G.node[n]["action"] = 0
            self.G.node[n]["n_signal"] = 0 #状態の初期値は0


    def run(self):
        rand = True
        nodes = self.G.nodes()

        for i in range(self.__n_round):
            if i % 100 == 0:
                print('iter : '+str(i))

            # 全エージェントが同期的に行動選択
            if i == len(self.__payoff_matrix)*5: #初期のランダム行動
                rand=False

            # 全てのエージェントが行動選択
            for n in nodes:
                self.G.node[n]["action"] = self.G.node[n]["agent"].act(self.G.node[n]["n_signal"], random=rand, reduction=True) # for Q Leaning, FAQ
                # self.G.node[n]["action"] = self.G.node[n]["agent"].act(self.G.node[n]["n_signal"], random=rand)  # for others, change action
                self.agent_action_table[n][i] = self.G.node[n]["action"]

            hoge = 0
            fuga = 0
            # 報酬計算&Q値更新
            for n in nodes:
                neighbors = self.G.neighbors(n)
                n_action = self.G.node[n]["action"]
                n_reward = 0
                n_signal = 0
                for ne in neighbors:
                    hoge += 1
                    ne_action = self.G.node[ne]["action"]

                    # noisyなsignal
                    noise = np.random.rand()
                    is_noise = (self.__p_noise >= noise) # Trueだったらnoise環境
                    if is_noise:
                        fuga += 1
                    if ne_action==0 or ne_action==2:
                        if not is_noise: # 正しく伝えわる
                            n_signal += 1
                    else:
                        if is_noise: # 間違って伝わる
                            n_signal += 1

                    n_reward += self.__payoff_matrix[ne_action][n_action]

                self.G.node[n]["reward"] = n_reward
                self.agent_payoff_table[n][i] = n_reward/len(neighbors)
                self.G.node[n]["agent"].update_q(self.G.node[n]["n_signal"], n_reward)
                # self.G.node[n]["agent"].update(self.G.node[n]["n_signal"], n_reward, n_action) # for SARSA
                self.G.node[n]["n_signal"] = n_signal
            print(fuga/hoge)

# allはsig


if __name__ == "__main__":
    # pとかグラフは引数にしよう
    RESULT_DIR = "../../../results/Q_reduction/noise/0.5/"
    for n in all_:
        RESULT_NAME = RESULT_DIR+n+'_q_reduc_signal_complete_n0.5'
        W = synchro_world_signal(100, 10000, eval(n)(), nx.barabasi_albert_graph, Q_Learning_Agent, p_noise=0.5) #妥当なエージェント数はいくつか
        W.run()
        W.save(RESULT_NAME)
        print('save done!!')
