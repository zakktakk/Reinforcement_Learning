#-*- coding: utf-8 -*-
# author : Takuro Yamazaki
# description : 隣接エージェントの行動が観察可能な場合

# TODO coop側の人しかsignal出せないかつ，signal出す出さないは選択可能，signalはコストがかかる -> バランス取らないと初期の行動で行動割合の偏りが生じるぞ
# TODO 全体の状況が見える場合、近傍の状況が見える場合


"""path setting"""
import sys
sys.path.append("../")

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

"""self made library"""
from .synchro_world import synchro_world


class synchro_world_observable(synchro_world):
    def __init__(self, n_agent, n_round, payoff_mat, nwk_alg, rl_alg,episodes=5,
                 altered_mat=None, share_rate=None, nwk_param=None, rl_param=None,
                 rewire_mode=None, rewire_interval=0, p_noise=-1):
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
        :param p_noise: probablity of noise in observable information
        """
        super().__init__(n_agent, n_round, payoff_mat, nwk_alg, rl_alg, episodes, altered_mat, share_rate, nwk_param, rl_param,
                 rewire_mode, rewire_interval)
        self.p_noise = p_noise


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

        for n in self.G.nodes():
            neighbors = self.G.neighbors(n)
            agent = self.rl_alg(n, np.arange(len(neighbors)+1), self.payoff_mat.index, **rl_param)
            self.G.node[n]["agent"] = agent
            self.G.node[n]["action"] = 0
            self.G.node[n]["n_signal"] = 0 # initial state is zero


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
                self.G.node[n]["action"] = self.G.node[n]["agent"].act(self.G.node[n]["n_signal"], random=rand)
                if self.G.node[n]["action"] == "c": coop_num += 1

            self.coop_per_df[0][i] = coop_num / self.n_agent

            # 報酬計算&Q値更新
            reward_sum = 0
            for n in nodes:
                neighbors = self.G.neighbors(n)
                n_action = self.G.node[n]["action"]

                # reward
                n_reward = 0
                n_reward_min = 10000
                n_reward_argmin = 0

                # signal
                n_signal = 0

                for ne in neighbors:
                    # -------- basic action
                    ne_action = self.G.node[ne]["action"]
                    n_ne_payoff = self.payoff_mat[ne_action][n_action]
                    n_reward += n_ne_payoff

                    # -------- noisyなsignal
                    noise = np.random.rand()
                    is_noise = (self.p_noise > noise)  # Trueだったらnoise
                    if ne_action == "c":
                        if not is_noise:
                            n_signal += 1
                    else:
                        if is_noise:
                            n_signal += 1

                    # -------- rewire
                    if is_rewire:
                        if n_reward_min > n_ne_payoff:
                            n_reward_min = n_ne_payoff
                            n_reward_argmin = ne

                        elif n_reward_min == n_ne_payoff and (np.random.rand() < 0.5):
                            n_reward_argmin = ne

                        rewire_lst.append((n, n_reward_argmin))

                self.G.node[n]["reward"] = n_reward
                reward_sum += n_reward / len(neighbors)
                self.G.node[n]["n_signal"] = n_signal
                self.G.node[n]["agent"].update(self.G.node[n]["n_signal"], n_reward)
                #self.G.node[n]["agent"].update(self.G.node[n]["n_signal"], n_reward, n_action) # for SARSA

            if self.share_rate is not None:
                self.update_q()
                self.share_q()

            if is_rewire:
                self.rewire_mode(rewire_lst)

            self.payoff_df[0][i] = reward_sum / self.n_agent


    def save(self, f_name: str, other=None) -> None:
        """実験結果の保存
        :param f_name: 出力ファイル名
        :param other: その他の条件
        :return: None
        """
        self.save_average_reward(f_name + "_reward.csv")
        self.save_average_coop(f_name + "_coop.csv")
        self.save_meta_info(f_name + "_meta.txt", other)


    def get_result(self, f_name: str, other=None):
        """
        :param f_name: 出力ファイル名
        :param other: その他の条件
        :return: q値のテーブル、協調確率のテーブル、獲得報酬のテーブル
        """
        self.save_meta_info(f_name+"_meta.txt", other)
        return self.coop_per_df, self.payoff_df