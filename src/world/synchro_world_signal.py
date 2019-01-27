#-*- coding: utf-8 -*-
# author : Takuro Yamazaki
# description : チープトークを含む場合

# TODO coop側の人しかsignal出せないかつ，signal出す出さないは選択可能，signalはコストがかかる -> バランス取らないと初期の行動で行動割合の偏りが生じるぞ
# TODO 全体の助教が見える場合、近傍の状況が見える場合


"""path setting"""
import sys
sys.path.append("../")

"""util libraries"""
import numpy as np
import networkx as nx
from tqdm import tqdm

"""visualize"""
import matplotlib as mpl
mpl.use('tkagg')

import matplotlib.pyplot as plt
plt.style.use('ggplot')

"""self made library"""
from .synchro_world import synchro_world


class synchro_world_signal(synchro_world):
    def __init__(self, n_agent, n_round, payoff_func, nwk_alg, rl_alg,
                 altered_func=None, share_rate=None, nwk_param=None, rl_param=None, p_noise=-1):
        """
        :param n_agent: agent num
        :param n_round: round num
        :param payoff_func: payoff matrix
        :param nwk_alg: network algorithm
        :param rl_alg: reinforcement algorithm
        :param altered_func: payoff matrix that is used when accident occur
        :param share_rate: q value share rate. if None never sharing
        :param nwk_param: network parameter
        :param rl_param: reinforcement learning parameter
        :param rewire_mode: rewire method or on rewiring
        :param rewire_interval: edge rewiring interval, if 0, no rewiring
        :param p_noise: probablity of noise in observable information
        """
        super().__init__(n_agent, n_round, payoff_func, nwk_alg, rl_alg, altered_func, share_rate, nwk_param, rl_param)
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

        if self.is_aca:
            reguralize_value = np.average(list(self.G.degree().values()))

        for n in self.G.nodes():
            neighbors = self.G.neighbors(n)
            if self.is_aca:
                agent = self.rl_alg(n, np.arange(len(neighbors)+1), ['cs', 'c', 'ds', 'd'], reguralize_value, **rl_param)
            else:
                agent = self.rl_alg(n, np.arange(len(neighbors)+1), ['cs', 'c', 'ds', 'd'], **rl_param)

            self.G.node[n]["agent"] = agent
            self.G.node[n]["action"] = 0
            self.G.node[n]["n_signal"] = 0 # initial state is zero


    def run(self) -> None:
        """run simulation
        """
        rand = True
        nodes = self.G.nodes()

        for i in tqdm(range(self.n_round)):
            # 全エージェントが同期的に行動選択
            if i == int(self.n_round * 0.01): rand = False

            # payoff matの更新
            if (self.altered_func is not None) and (i == self.n_round * 0.5): self.change_payoff_metrix()

            # 全てのエージェントが行動選択
            coop_num = 0
            for n in nodes:
                self.G.node[n]["action"] = self.G.node[n]["agent"].act(self.G.node[n]["n_signal"], random=rand)

                if self.G.node[n]["action"] == "c" or self.G.node[n]["action"] == "cs":
                    coop_num += 1

            self.coop_per_df[0][i] = coop_num / self.n_agent

            # 報酬計算&Q値更新
            reward_sum = 0
            for n in nodes:
                neighbors = self.G.neighbors(n)
                n_action = self.G.node[n]["action"]

                # coop
                ne_coop_num = 0
                # signal
                n_signal = 0

                for ne in neighbors:
                    # -------- basic action
                    ne_action = self.G.node[ne]["action"]
                    if ne_action == "c" or ne_action == "cs": ne_coop_num += 1

                    # -------- noisyなsignal
                    noise = np.random.rand()
                    is_noise = (self.p_noise > noise)  # Trueだったらnoise
                    if ne_action == "cs" or ne_action=="ds":
                        if not is_noise:
                            n_signal += 1
                    else:
                        if is_noise:
                            n_signal += 1

                if n_action == "c" or n_action == "cs": n_reward = self.payoff_func[0] * ne_coop_num + self.payoff_func[1]
                else: n_reward = self.payoff_func[2] * ne_coop_num + self.payoff_func[3]

                self.G.node[n]["reward"] = n_reward
                reward_sum += n_reward / len(neighbors)
                self.G.node[n]["n_signal"] = n_signal
                self.G.node[n]["agent"].update(self.G.node[n]["n_signal"], n_reward) # 今状態は0だけ
                #self.G.node[n]["agent"].update(self.G.node[n]["n_signal"], n_reward, n_action) # for SARSA


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
