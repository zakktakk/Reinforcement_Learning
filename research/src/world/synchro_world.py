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

class synchro_world(object):
    def __init__(self, n_agent, n_round, payoff_matrix, network_alg, rl_alg):
        self.__n_agent = n_agent
        self.__n_round = n_round
        self.__game_name, self.__payoff_matrix = payoff_matrix
        self.__network_alg = network_alg
        self.__rl_alg = rl_alg
        self.__create_network()
        self.agent_action_table = pd.DataFrame(np.zeros((n_round, n_agent)))
        self.agent_payoff_table = pd.DataFrame(np.zeros((n_round, n_agent)))


    def __create_network(self) -> None:
        """
        :description ネットワークモデルの定義
        :param n_agent : エージェントの数
        """
        self.G = self.__network_alg()  # change network
        self.n_edges = self.G.size() # number of edges

        for n in self.G.nodes():
            neighbors = self.G.neighbors(n)
            agent = self.__rl_alg(n, np.array([0]), self.__payoff_matrix.index)
            self.G.node[n]["agent"] = agent
            self.G.node[n]["action"] = 0



    def run(self):
        rand = True
        nodes = self.G.nodes()
        for i in range(self.__n_round):
            if i % 1000 == 0: # 途中経過の出力
                print('iter : '+str(i))

            # 全エージェントが同期的に行動選択
            if i == len(list(self.__payoff_matrix))*5:  # 初期のランダム行動
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
                for ne in neighbors:
                    ne_action = self.G.node[ne]["action"]
                    n_reward += self.__payoff_matrix[ne_action][n_action]

                self.G.node[n]["reward"] = n_reward
                self.agent_payoff_table[n][i] = n_reward/len(neighbors)
                self.G.node[n]["agent"].update(0, n_reward) # 今状態は0だけ
                # self.G.node[n]["agent"].update(0, n_reward, n_action) # for SARSA


    def __save_meta_info(self, f_name:str, other=None) -> None:
        """実験条件の保存
        :param f_name: 出力ファイル名
        :param other: その他の条件を出力する
        """
        with open(f_name, "w") as f:
            f.write("繰り返し回数 : "+str(self.__n_round)+"\n")
            f.write("エージェント数 : "+str(self.__n_agent)+"\n")
            f.write("エッジ数 : "+str(self.n_edges)+"\n")
            f.write("ネットワーク種類 : "+self.__network_alg.__name__+"\n")
            f.write("利得行列 : "+self.__game_name+"\n")
            f.write("強化学習アルゴリズム : "+self.__rl_alg.__name__+"\n")
            if other is not None:
                f.write("その他の条件 : "+other)


    def __save_average_reward(self, f_name: str) -> None:
        """各ステップでの平均報酬を保存
        :param f_name: 出力ファイル名
        :return: None
        """
        self.agent_payoff_table.apply(lambda x:np.mean(x), axis=1).to_csv(f_name, index=False)


    def __save_action_table(self, f_name: str) -> None:
        """agentの行動を保存
        :param f_name: 出力ファイル名
        :return: None
        """
        self.agent_action_table.to_csv(f_name, index=False)

    def __save_graph_pickle(self, f_name: str) -> None:
        """グラフをpickleで保存
        :param f_name: 出力ファイル名
        :return: None
        """
        nx.write_gpickle(self.G, f_name)


    def save(self, f_name: str, other=None) -> None:
        """実験結果の保存
        :param f_name: 出力ファイル名
        :param other: その他の条件
        :return: None
        """
        self.__save_action_table(f_name+"_action_table.csv")
        self.__save_average_reward(f_name+"_ave.csv")
        self.__save_graph_pickle(f_name+"_g.gpickle")
        self.__save_meta_info(f_name+"_meta.txt", other)
