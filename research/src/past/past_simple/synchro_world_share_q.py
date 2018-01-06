# -*- coding: utf-8 -*-
# author : Takuro Yamazaki
# description : Q値を周りのエージェントと共有した時の行動


"""path setting"""
import sys

sys.path.append("../")

"""util libraries"""
import numpy as np
import pandas as pd

"""visualize"""
# ディスプレイ表示させない
import matplotlib as mpl

mpl.use('tkagg')

import matplotlib.pyplot as plt

plt.style.use('ggplot')

"""self made library"""
from research.src.world.synchro_world_simple import synchro_world


# share rate分相手の意見を聞く

class synchro_world_share_q(synchro_world):
    def __init__(self, n_agent, n_round, payoff_matrix, network_alg, rl_alg, share_rate):
        self.n_agent = n_agent
        self.n_round = n_round
        self.game_name, self.payoff_matrix = payoff_matrix
        self.network_alg = network_alg
        self.rl_alg = rl_alg
        self.share_rate = share_rate
        self.payoff_table = pd.DataFrame(np.zeros((n_round, 1)))
        self.coop_per_table = pd.DataFrame(np.zeros((n_round, 1)))
        self.agent_q_table = pd.DataFrame(np.zeros((n_agent, len(self.payoff_matrix.index))))
        self.create_network()


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
            self.agent_q_table.iloc[n] = np.array(agent.q_table[0])
            self.G.node[n]["agent"] = agent
            self.G.node[n]["action"] = 0


    def _update_q(self):
        for n in self.G.nodes():
            self.agent_q_table.iloc[n] = np.array(self.G.node[n]["agent"].q_table)[:, 0]


    def _share_q(self):
        for n in self.G.nodes():
            neighbors_q = self.agent_q_table.iloc[self.G.neighbors(n)].mean(axis=0)
            self.agent_q_table.iloc[n] = self.agent_q_table.iloc[n] * (1 - self.share_rate) + self.share_rate * neighbors_q
            self.G.node[n]["agent"].q_table.iloc[0] = self.agent_q_table.iloc[n][0]
            self.G.node[n]["agent"].q_table.iloc[1] = self.agent_q_table.iloc[n][1]


    def run(self):
        rand = True
        nodes = self.G.nodes()
        for i in range(self.n_round):
            if i % 1000 == 0:  # 途中経過の出力
                print('iter : ' + str(i))

            # 全エージェントが同期的に行動選択
            if i == len(list(self.payoff_matrix)) * 5:  # 初期のランダム行動
                rand = False

            rewards = []
            actions = []
            # 全てのエージェントが隣接エージェントと対戦＆更新
            for n1 in nodes:
                n2 = np.random.choice(self.G.neighbors(n1))

                n1_action = self.G.node[n1]["agent"].act(0, random=rand)
                n2_action = self.G.node[n2]["agent"].act(0, random=rand)

                n1_reward = self.payoff_matrix[n2_action][n1_action]
                n2_reward = self.payoff_matrix[n1_action][n2_action]

                rewards.append(n1_reward)
                rewards.append(n2_reward)

                actions.append(n1_action)
                actions.append(n2_action)

                self.G.node[n1]["agent"].update(0, n1_reward)  # 今状態は0だけ
                # self.G.node[n1]["agent"].update(0, n1_reward, n1_action) # for SARSA

                self.G.node[n2]["agent"].update(0, n2_reward)
                # self.G.node[n2]["agent"].update(0, n2_reward, n2_action) # for SARSA

            self._update_q()
            self._share_q()

            actions = np.array(actions)
            self.payoff_table[0][i] = np.mean(rewards)
            self.coop_per_table[0][i] = len(actions[actions == "c"]) / self.n_agent / 2
            # TODO save Q value
