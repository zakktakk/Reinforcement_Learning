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

from agent import Actor_Critic_Agent as aca

"""util libraries"""
import numpy as np
import networkx as nx
import pandas as pd
from tqdm import tqdm

"""visualize"""
import matplotlib as mpl
mpl.use('tkagg')

import matplotlib.pyplot as plt
plt.style.use('ggplot')


class synchro_world(object):
    def __init__(self, n_agent, n_round, payoff_mat, nwk_alg, rl_alg, episodes=5,
                 altered_mat=None, share_rate=None, nwk_param=None, rl_param=None, rewire_mode=None, rewire_interval=0):
        """
        :param n_agent: agent num
        :param n_round: round num
        :param payoff_mat: payoff matrix
        :param nwk_alg: network algorithm
        :param rl_alg: reinforcement algorithm
        :param altered_mat: payoff matrix that is used when accident occur
        :param share_rate: q value share rate. if None never sharing
        :param nwk_param: network parameter
        :param rl_param: reinforcement learning parameter
        :param rewire_mode: rewire method or on rewiring
        :param rewire_interval: edge rewiring interval, if 0, no rewiring
        """
        self.n_agent = n_agent
        self.n_round = n_round
        self.game_name, self.payoff_mat = payoff_mat
        self.rl_alg = rl_alg

        if rl_alg == aca.Actor_Critic_Agent:
            self.skip_q = True
        else:
            self.skip_q = False

        self.nwk_alg = nwk_alg
        self.episodes = episodes
        self.is_prepared_nwk = isinstance(nwk_alg, str)

        # set rewiring method
        if rewire_mode == "random":
            self.rewire_mode = self.random_rewire
        elif rewire_mode == "neighbor":
            self.rewire_mode = self.neighbors_neighbor
        elif rewire_mode == "preference":
            self.rewire_mode = self.preference_attachment
        else:
            self.rewire_mode = None

        self.rewire_interval = rewire_interval if self.rewire_mode is not None else 0

        # set altered matrix
        self.altered_mat = altered_mat
        # set q value share rate
        self.share_rate = share_rate

        # set dataframe for saving data
        self.payoff_df = pd.DataFrame(np.zeros((n_round, 1)))
        self.coop_per_df = pd.DataFrame(np.zeros((n_round, 1)))
        self.q_df = pd.DataFrame(np.zeros((n_round, len(self.payoff_mat.index))))
        self.agent_q_df = pd.DataFrame(np.zeros((n_agent, len(self.payoff_mat.index))))

        # create networked simulation envirionment
        nwk_param = {} if nwk_param is None else nwk_param
        rl_param = {} if rl_param is None else rl_param
        self.create_nwk(nwk_param, rl_param)


    def create_nwk(self, nwk_param, rl_param) -> None:
        """create agent network
        :param nwk_param: network parameter
        :param rl_param: reinforcement algorithm parameter
        :return: None
        """

        if self.is_prepared_nwk:
            self.G = nx.read_gpickle(self.nwk_alg)
        else:
            self.G = self.nwk_alg(**nwk_param)

        if self.skip_q:
            reguralize_value = np.average(list(self.G.degree().values()))

        for n in self.G.nodes():
            if self.skip_q:
                agent = self.rl_alg(n, np.array([0]), self.payoff_mat.index, reguralize_value, **rl_param)
            else:
                agent = self.rl_alg(n, np.array([0]), self.payoff_mat.index, **rl_param)
            self.G.node[n]["agent"] = agent
            self.G.node[n]["action"] = 0

    def change_payoff_metrix(self):
        print("matrix altered!!")
        self.payoff_mat = self.altered_mat


    def update_q(self):
        for n in self.G.nodes():
            self.agent_q_df.iloc[n] = np.array(self.G.node[n]["agent"].q_df)[:, 0]


    def share_q(self):
        for n in self.G.nodes():
            neighbors_q = self.agent_q_df.iloc[self.G.neighbors(n)].mean(axis=0)
            self.agent_q_df.iloc[n] = self.agent_q_df.iloc[n] * (1 - self.share_rate) + self.share_rate * neighbors_q

            for i in range(len(self.payoff_mat.index)):
                self.G.node[n]["agent"].q_df.iloc[i] = self.agent_q_df.iloc[n][i]

    def write_q_val(self, i):
        neighbor_num = len(self.G.neighbors(0))
        q_val = self.G.node[0]["agent"].q_df.as_matrix() / neighbor_num

        for n in self.G.nodes()[1:]:
            neighbor_num = len(self.G.neighbors(n))
            q_val += self.G.node[n]["agent"].q_df.as_matrix() / neighbor_num

        q_val /= self.n_agent

        for j in range(len(self.payoff_mat.index)):
            self.q_df.iloc[i][j] = q_val[j][0]


    def run(self) -> None:
        """run simulation
        """
        rand = True
        nodes = self.G.nodes()

        for i in tqdm(range(self.n_round)):
            is_rewire = (self.rewire_interval > 0) and (i % self.rewire_interval == 0)
            rewire_lst = []

            # 全エージェントが同期的に行動選択
            if i == len(list(self.payoff_mat))*5: rand = False

            # payoff matの更新
            if (self.altered_mat is not None) and (i == self.n_round * 0.5): self.change_payoff_metrix()

            # 全てのエージェントが行動選択
            coop_num = 0
            for n in nodes:
                self.G.node[n]["action"] = self.G.node[n]["agent"].act(0, random=rand)  # for Q Leaning, FAQ

                if self.G.node[n]["action"] == "c": coop_num += 1

            self.coop_per_df[0][i] = coop_num / self.n_agent

            # 報酬計算&Q値更新
            reward_sum = 0
            for n in nodes:
                neighbors = self.G.neighbors(n)
                n_action = self.G.node[n]["action"]
                n_reward = 0
                n_reward_min = 10000
                n_reward_argmin = 0

                for ne in neighbors:
                    ne_action = self.G.node[ne]["action"]
                    n_ne_payoff = self.payoff_mat[ne_action][n_action]
                    n_reward += n_ne_payoff
                    if is_rewire: # if rewire occured
                        if n_reward_min > n_ne_payoff:
                            n_reward_min = n_ne_payoff
                            n_reward_argmin = ne
                        elif n_reward_min == n_ne_payoff and (np.random.rand() < 0.5):
                            n_reward_argmin = ne

                        rewire_lst.append((n, n_reward_argmin))

                self.G.node[n]["reward"] = n_reward
                reward_sum += n_reward / len(neighbors)
                self.G.node[n]["agent"].update(0, n_reward) # 今状態は0だけ
                #self.G.node[n]["agent"].update(0, n_reward, n_action) # for SARSA

            # save average q value of each round
            if not self.skip_q:
                self.write_q_val(i)

            if self.share_rate is not None:
                self.update_q()
                self.share_q()

            if is_rewire:
                self.rewire_mode(rewire_lst)

            self.payoff_df[0][i] = reward_sum / self.n_agent


    def save_meta_info(self, f_name:str, other=None) -> None:
        """実験条件の保存
        :param f_name: 出力ファイル名
        :param other: その他の条件を出力する
        """
        with open(f_name, "w") as f:
            f.write("繰り返し回数 : "+str(self.n_round)+"\n")
            f.write("エージェント数 : "+str(self.n_agent)+"\n")
            f.write("エッジ数 : "+str(self.G.size())+"\n")

            if self.is_prepared_nwk:
                f.write("ネットワーク種類 : " + self.nwk_alg + "\n")
            else:
                f.write("ネットワーク種類 : "+self.nwk_alg.__name__+"\n")

            f.write("利得行列 : "+self.game_name+"\n")
            f.write("強化学習アルゴリズム : "+self.rl_alg.__name__+"\n")
            if other is not None:
                f.write("その他の条件 : "+other)


    def save_average_reward(self, f_name: str) -> None:
        """各ステップでの平均報酬を保存
        :param f_name: 出力ファイル名
        :return: None
        """
        self.payoff_df.to_csv(f_name, index=False, header=False)


    def save_average_coop(self, f_name: str) -> None:
        """各ステップでの協調行動割合を保存
        :param f_name: 出力ファイル名
        :return: None
        """
        self.coop_per_df.to_csv(f_name, index=False, header=False)


    def save_q_value(self, f_name: str) -> None:
        """q値を保存
        :param f_name: 出力ファイル名
        :return: None
        """
        self.q_df.to_csv(f_name, index=False, header=False)


    def save(self, f_name: str, other=None) -> None:
        """実験結果の保存
        :param f_name: 出力ファイル名
        :param other: その他の条件
        :return: None
        """
        self.save_average_reward(f_name+"_reward.csv")
        self.save_average_coop(f_name+"_coop.csv")
        self.save_q_value(f_name+"_q.csv")
        self.save_meta_info(f_name+"_meta.txt", other)


    def get_result(self, f_name: str, other=None):
        """
        :param f_name: 出力ファイル名
        :param other: その他の条件
        :return: q値のテーブル、協調確率のテーブル、獲得報酬のテーブル
        """
        self.save_meta_info(f_name+"_meta.txt", other)
        return self.q_df, self.coop_per_df, self.payoff_df


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
