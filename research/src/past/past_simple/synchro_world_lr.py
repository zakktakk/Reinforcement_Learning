# -*- coding: utf-8 -*-
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
# ディスプレイ表示させない
import matplotlib as mpl

mpl.use('tkagg')

import matplotlib.pyplot as plt

plt.style.use('ggplot')


class synchro_world_lr(object):
    def __init__(self, n_agent, n_round, payoff_matrix, network_alg, rl_alg, gamma):
        self.n_agent = n_agent
        self.n_round = n_round
        self.game_name, self.payoff_matrix = payoff_matrix
        self.network_alg = network_alg
        self.gamma = gamma
        print(self.gamma)
        self.rl_alg = rl_alg
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

            actions = np.array(actions)
            self.payoff_table[0][i] = np.mean(rewards)
            self.coop_per_table[0][i] = len(actions[actions == "c"]) / self.n_agent / 2
            # TODO save Q value

    def __save_meta_info(self, f_name: str, other=None) -> None:
        """実験条件の保存
        :param f_name: 出力ファイル名
        :param other: その他の条件を出力する
        """
        with open(f_name, "w") as f:
            f.write("繰り返し回数 : " + str(self.n_round) + "\n")
            f.write("エージェント数 : " + str(self.n_agent) + "\n")
            f.write("エッジ数 : " + str(self.n_edges) + "\n")
            f.write("ネットワーク種類 : " + self.network_alg.__name__ + "\n")
            f.write("利得行列 : " + self.game_name + "\n")
            f.write("強化学習アルゴリズム : " + self.rl_alg.__name__ + "\n")
            if other is not None:
                f.write("その他の条件 : " + other)

    def __save_average_reward(self, f_name: str) -> None:
        """各ステップでの平均報酬を保存
        :param f_name: 出力ファイル名
        :return: None
        """
        self.payoff_table.to_csv(f_name, index=False)

    def __save_average_coop(self, f_name: str) -> None:
        """各ステップでの協調行動割合を保存
        :param f_name: 出力ファイル名
        :return: None
        """
        self.coop_per_table.to_csv(f_name, index=False)

    def save(self, f_name: str, other=None) -> None:
        """実験結果の保存
        :param f_name: 出力ファイル名
        :param other: その他の条件
        :return: None
        """
        self.__save_average_reward(f_name + "_reward.csv")
        self.__save_average_coop(f_name + "_coop.csv")
        self.__save_meta_info(f_name + "_meta.txt", other)