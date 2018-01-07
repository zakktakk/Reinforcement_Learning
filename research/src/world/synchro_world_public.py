#-*- coding: utf-8 -*-
# author : Takuro Yamazaki
# description : 公共の情報が参照可能な場合


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


class synchro_world_public(synchro_world):
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
            if self.is_aca:
                agent = self.rl_alg(n, np.arange(self.n_agent), ["c", "d"], reguralize_value, **rl_param)
            else:
                agent = self.rl_alg(n, np.arange(self.n_agent), ["c", "d"], **rl_param)
            self.G.node[n]["agent"] = agent
            self.G.node[n]["action"] = 0


    def run(self) -> None:
        """run simulation
        """
        rand = True
        nodes = self.G.nodes()
        prev_public_coop_num = 0

        for i in tqdm(range(self.n_round)):
            # 全エージェントが同期的に行動選択
            if i == int(self.n_round * 0.01): rand = False

            # payoff matの更新
            if (self.altered_func is not None) and (i == self.n_round * 0.5): self.change_payoff_metrix()

            # 全てのエージェントが行動選択
            coop_num = 0
            public_coop_num = 0
            for n in nodes:
                self.G.node[n]["action"] = self.G.node[n]["agent"].act(prev_public_coop_num, random=rand)  # for Q Leaning, FAQ

                # -------- noisyな観測
                noise = np.random.rand()
                is_noise = (self.p_noise > noise)

                if self.G.node[n]["action"] == "c":
                    coop_num += 1
                    if not is_noise:
                        public_coop_num += 1
                else:
                    if is_noise:
                        public_coop_num += 1


            self.coop_per_df[0][i] = coop_num / self.n_agent

            # 報酬計算&Q値更新
            reward_sum = 0
            for n in nodes:
                neighbors = self.G.neighbors(n)
                n_action = self.G.node[n]["action"]

                # reward
                n_reward = 0
                # coop
                ne_coop_num = 0

                for ne in neighbors:
                    # -------- basic action
                    ne_action = self.G.node[ne]["action"]
                    if ne_action == "c": ne_coop_num += 1

                if n_action == "c": n_reward = self.payoff_func[0] * ne_coop_num + self.payoff_func[1]
                else: n_reward = self.payoff_func[2] * ne_coop_num + self.payoff_func[3]

                self.G.node[n]["reward"] = n_reward
                reward_sum += n_reward
                self.G.node[n]["agent"].update(public_coop_num, n_reward)
                #self.G.node[n]["agent"].update(0, n_reward, n_action) # for SARSA
                prev_public_coop_num = public_coop_num


            if self.share_rate is not None:
                self.update_q()
                self.share_q()

            self.payoff_df[0][i] = reward_sum / self.n_agent


    def save(self, f_name: str, other=None) -> None:
        """実験結果の保存
        :param f_name: 出力ファイル名
        :param other: その他の条件
        :return: None
        """
        self.save_average_reward(f_name+"_reward.csv")
        self.save_average_coop(f_name+"_coop.csv")
        self.save_meta_info(f_name+"_meta.txt", other)

    def get_result(self, f_name: str, other=None):
        """
        :param f_name: 出力ファイル名
        :param other: その他の条件
        :return: q値のテーブル、協調確率のテーブル、獲得報酬のテーブル
        """
        self.save_meta_info(f_name+"_meta.txt", other)
        return self.coop_per_df, self.payoff_df
