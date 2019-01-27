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

from agent import WoLF_PHC_Agent as wolf
from agent import Actor_Critic_Agent as aca
from agent import Actor_Critic_Agent_TD as acaTD
from agent import SARSA_Agent as sarsa
from agent import SARSA_Agent_TD as sarsaTD

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
    def __init__(self, n_agent, n_round, payoff_func, nwk_alg, rl_alg,
                 altered_func=None, share_rate=None, nwk_param=None, rl_param=None):
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
        """
        self.n_agent = n_agent
        self.n_round = n_round
        self.payoff_func = payoff_func
        self.rl_alg = rl_alg

        self.is_aca = (rl_alg == aca.Actor_Critic_Agent or rl_alg == acaTD.Actor_Critic_Agent_TD)
        self.is_sarsa = (rl_alg == sarsa.SARSA_Agent or rl_alg == sarsaTD.SARSA_Agent_TD)
        self.is_wolf = (rl_alg == wolf.WoLF_PHC_Agent)

        self.nwk_alg = nwk_alg
        self.is_prepared_nwk = isinstance(nwk_alg, str)

        # set altered matrix
        self.altered_func = altered_func
        # set q value share rate
        self.share_rate = share_rate

        # set dataframe for saving data
        self.payoff_df = pd.DataFrame(np.zeros((n_round, 1)))
        self.coop_per_df = pd.DataFrame(np.zeros((n_round, 1)))
        self.q_df = pd.DataFrame(np.zeros((n_round, 2)))
        self.agent_q_df = pd.DataFrame(np.zeros((n_agent, 2)))
        self.history = pd.DataFrame(np.zeros((n_round, n_agent)))

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

        if self.is_aca:
            reguralize_value = np.average(list(self.G.degree().values()))

        for n in self.G.nodes():
            if self.is_aca:
                agent = self.rl_alg(n, np.array([0]), ["c", "d"], reguralize_value, **rl_param)
            else:
                agent = self.rl_alg(n, np.array([0]), ["c", "d"], **rl_param)
            self.G.node[n]["agent"] = agent
            self.G.node[n]["action"] = 0


    def change_payoff_metrix(self):
        print("matrix altered!!")
        self.payoff_func = self.altered_func


    def update_q(self):
        for n in self.G.nodes():
            self.agent_q_df.iloc[n] = np.array(self.G.node[n]["agent"].q_df)[:, 0]


    def share_q(self):
        for n in self.G.nodes():
            neighbors_q = self.agent_q_df.iloc[self.G.neighbors(n)].mean(axis=0)
            self.agent_q_df.iloc[n] = self.agent_q_df.iloc[n] * (1 - self.share_rate) + self.share_rate * neighbors_q

            for i in range(2):
                self.G.node[n]["agent"].q_df.iloc[i] = self.agent_q_df.iloc[n][i]

    def write_pi_val(self, i):
        neighbor_num = len(self.G.neighbors(0))
        pi_val = self.G.node[0]["agent"].pi_df.as_matrix() / neighbor_num

        for n in self.G.nodes()[1:]:
            neighbor_num = len(self.G.neighbors(n))
            pi_val += self.G.node[n]["agent"].pi_df.as_matrix() / neighbor_num

        pi_val /= self.n_agent

        for j in range(2):
            self.q_df.iloc[i][j] = pi_val[j][0]


    def write_q_val(self, i):
        neighbor_num = len(self.G.neighbors(0))
        q_val = self.G.node[0]["agent"].q_df.as_matrix() / neighbor_num

        for n in self.G.nodes()[1:]:
            neighbor_num = len(self.G.neighbors(n))
            q_val += self.G.node[n]["agent"].q_df.as_matrix() / neighbor_num

        q_val /= self.n_agent

        for j in range(2):
            self.q_df.iloc[i][j] = q_val[j][0]


    def write_p_val(self, i):
        neighbor_num = len(self.G.neighbors(0))
        p_val = self.G.node[0]["agent"].p_df.as_matrix() / neighbor_num

        for n in self.G.nodes()[1:]:
            neighbor_num = len(self.G.neighbors(n))
            p_val += self.G.node[n]["agent"].p_df.as_matrix() / neighbor_num

        p_val /= self.n_agent

        for j in range(len(["c", "d"])):
            self.q_df.iloc[i][j] = p_val[j][0]


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
                self.G.node[n]["action"] = self.G.node[n]["agent"].act(0, random=rand)
                self.history[n][i] = self.G.node[n]["action"]

                if self.G.node[n]["action"] == "c": coop_num += 1

            self.coop_per_df[0][i] = coop_num / self.n_agent

            # 報酬計算&Q値更新
            reward_sum = 0
            for n in nodes:
                neighbors = self.G.neighbors(n)
                n_action = self.G.node[n]["action"]
                ne_coop_num = 0

                for ne in neighbors:
                    ne_action = self.G.node[ne]["action"]
                    if ne_action == "c": ne_coop_num += 1

                if n_action == "c": n_reward = self.payoff_func[0] * ne_coop_num + self.payoff_func[1]
                else: n_reward = self.payoff_func[2] * ne_coop_num + self.payoff_func[3]

                self.G.node[n]["reward"] = n_reward
                reward_sum += n_reward

                if self.is_sarsa: # if sarsa
                    self.G.node[n]["agent"].update(0, n_reward, n_action)
                else:
                    self.G.node[n]["agent"].update(0, n_reward)

            # save average q value of each round
            if self.is_aca:
                self.write_p_val(i)
            elif self.is_wolf:
                self.write_pi_val(i)
            else:
                self.write_q_val(i)

            if self.share_rate is not None:
                self.update_q()
                self.share_q()

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
