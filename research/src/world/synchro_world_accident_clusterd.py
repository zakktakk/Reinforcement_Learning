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



class synchro_world_accident_clustered(object):
    def __init__(self, n_agent, n_round, payoff_matrix, network_alg, rl_alg, after_matrix):
        self.n_agent = n_agent
        self.n_round = n_round
        self.game_name, self.payoff_matrix = payoff_matrix
        self.network_alg = network_alg
        self.after_matrix = after_matrix
        self.rl_alg = rl_alg
        self.create_network()
        self.payoff_table = pd.DataFrame(np.zeros((n_round, 1)))
        self.coop_per_table = pd.DataFrame(np.zeros((n_round, 1)))


    def create_network(self) -> None:
        """
        :description ネットワークモデルの定義
        :param n_agent : エージェントの数
        """
        self.G = nx.read_gpickle(self.network_alg)  # change network
        self.n_edges = self.G.size() # number of edges

        for n in self.G.nodes():
            neighbors = self.G.neighbors(n)
            agent = self.rl_alg(n, np.array([0]), self.payoff_matrix.index)
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


    def __save_meta_info(self, f_name:str, other=None) -> None:
        """実験条件の保存
        :param f_name: 出力ファイル名
        :param other: その他の条件を出力する
        """
        with open(f_name, "w") as f:
            f.write("繰り返し回数 : "+str(self.n_round)+"\n")
            f.write("エージェント数 : "+str(self.n_agent)+"\n")
            f.write("エッジ数 : "+str(self.n_edges)+"\n")
            f.write("ネットワーク種類 : "+self.network_alg+"\n")
            f.write("利得行列 : "+self.game_name+"\n")
            f.write("強化学習アルゴリズム : "+self.rl_alg.__name__+"\n")
            if other is not None:
                f.write("その他の条件 : "+other)


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
        self.__save_average_reward(f_name+"_reward.csv")
        self.__save_average_coop(f_name+"_coop.csv")
        self.__save_meta_info(f_name+"_meta.txt", other)