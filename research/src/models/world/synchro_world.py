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


"""setting"""
import sys
sys.path.append("../")

"""calculation"""
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
from payoff_matrix import *
from agent.FAQ_Agent import FAQ_Agent
from agent.SARSA_Agent import SARSA_Agent
from agent.WoLF_PHC_Agent import WoLF_PHC_Agent
from agent.Q_Learning_Agent import Q_Learning_Agent


class synchro_world(object):
    def __init__(self, n_agent, n_round, payoff_matrix):
        self.__n_agent = n_agent
        self.n_round = n_round
        self.__game_name, self.__payoff_matrix = payoff_matrix
        self.create_network(n_agent)
        self.agent_action_table = pd.DataFrame(np.zeros((n_round, n_agent)))
        self.agent_payoff_table = pd.DataFrame(np.zeros((n_round, n_agent)))


    def create_network(self, n_agent):
        """
        :description ネットワークモデルの定義
        :param n_agent : エージェントの数
        """
        self.G_alg, self.G = "ba", nx.barabasi_albert_graph(n_agent, 70)  # change network
        self.n_edges = 0
        self.rl_alg = "Q_Learning_Agent"  # change rl algorithm

        for n in self.G.nodes():
            neighbors = self.G.neighbors(n)
            self.n_edges += len(neighbors)
            agent = eval(self.rl_alg)(n, np.array(sorted(neighbors)), np.array(["0"]), self.__payoff_matrix)
            self.G.node[n]["agent"] = agent
            self.G.node[n]["action"] = 0


    def report_meta_info(self, f_name, other=None):
        """
        :param f_name: 出力ファイル名
        :param other: その他の条件を出力する
        """
        with open(f_name, "w") as f:
            f.write("繰り返し回数 : "+str(self.n_round)+"\n")
            f.write("エージェント数 : "+str(self.__n_agent)+"\n")
            f.write("エッジ数 : "+str(self.n_edges)+"\n")
            f.write("ネットワーク種類 : "+self.G_alg+"\n")
            f.write("利得行列 : "+self.__game_name+"\n")
            f.write("強化学習アルゴリズム : "+self.rl_alg+"\n")
            if other is not None:
                f.write("その他の条件 : "+other)


    def draw_network(self, f_name=None, w_degree=False):
        """
        :description Graphの可視化
        :param f_name : save file name
        :param w_degree : draw graph dependent on the node degree
        """
        # 行動によって色分けしてるけど別ので分類するなら変えよう
        color = ["r" if self.G.node[n]["action"] == 0 else "b" if self.G.node[n]["action"] == 1 else "g" for n in self.G.nodes()]

        if w_degree:
            d = nx.degree(self.G)
            nx.draw(self.G, pos=nx.circular_layout(self.G), nodelist=d.keys(),width=0.1,
                    alpha=0.5, node_color=color, node_size=[20+v*5 for v in d.values()])
        else:
            nx.draw(self.G, pos=nx.circular_layout(self.G), width=0.1 ,alpha=0.5, node_color=color)

        if f_name is None: # file名指定がなければただ表示
            plt.show()
        else:
            plt.savefig(f_name)


    def run(self):
        rand = True
        nodes = self.G.nodes()
        for i in range(self.n_round):
            if i % 1000 == 0: # 途中経過の出力
                print('iter : '+str(i))

            # 全エージェントが同期的に行動選択
            if i == len(list(self.__payoff_matrix))*5:  # 初期のランダム行動
                rand = False

            # 全てのエージェントが行動選択
            for n in nodes:
                self.G.node[n]["action"] = self.G.node[n]["agent"].act(0, random=rand, reduction=True)  # for Q Leaning, FAQ
                # self.G.node[n]["action"] = self.G.node[n]["agent"].act(0, random=rand)  # for others, change action
                self.agent_action_table[n][i] = self.G.node[n]["action"]

            # 報酬計算&Q値更新 **2回計算してるから非効率的やでー
            for n in nodes:
                neighbors = self.G.node[n]["agent"].get_neighbors()
                n_action = self.G.node[n]["action"]
                n_reward = 0
                for ne in neighbors:
                    ne_action = self.G.node[ne]["action"]
                    n_reward += self.__payoff_matrix[ne_action][n_action]

                self.G.node[n]["reward"] = n_reward
                self.agent_payoff_table[n][i] = n_reward/len(neighbors)
                self.G.node[n]["agent"].update(0, n_reward) # 今状態は0だけ
                # self.G.node[n]["agent"].update(0, n_reward, n_action) # for SARSA


    def save_average_reward(self, f_name):
        self.agent_payoff_table.apply(lambda x:np.mean(x), axis=1).to_csv(f_name, header=False, index=False)


    def save_coop_per(self, f_name):
        self.agent_action_table.apply(lambda x:len(x[x==0])/len(x), axis=1).to_csv(f_name, header=False, index=False)



if __name__ == "__main__":
    RESULT_DIR = "../../../results/Q_reduction/ba70"
    for n in all_:
        RESULT_NAME = RESULT_DIR+n+'_q_reduc_ba70'
        print(n)
        W = synchro_world(100, 100, eval(n)())  # 妥当なエージェント数はいくつか
        W.run()
        # W.save_average_reward(RESULT_NAME+"_ave.csv")
        # W.save_coop_per(RESULT_NAME+"_per.csv")
        # W.draw_network(f_name=RESULT_NAME+"_fig.png", w_degree=True)
        # W.report_meta_info(f_name=RESULT_NAME+"_meta.txt")
        print('save done!!')
