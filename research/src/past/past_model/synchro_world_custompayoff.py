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

"""self made library"""
from .synchro_world import synchro_world


class synchro_world(synchro_world):
    def __init__(self, n_agent, n_round, payoff_mat, nwk_alg, rl_alg,
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
        super().__init__(n_agent, n_round, payoff_mat, nwk_alg, rl_alg, altered_mat, share_rate, nwk_param, rl_param,
                         rewire_mode, rewire_interval)
        
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
                self.G.node[n]["action"] = self.G.node[n]["agent"].act(0, random=rand)

                if self.G.node[n]["action"] == "c": coop_num += 1

            self.coop_per_df[0][i] = coop_num / self.n_agent

            # 報酬計算&Q値更新
            reward_sum = 0
            for n in nodes:
                neighbors = self.G.neighbors(n)
                n_action = self.G.node[n]["action"]
                ne_coop = 0
                n_reward_min = 10000
                n_reward_argmin = 0

                for ne in neighbors:
                    ne_action = self.G.node[ne]["action"]
                    if ne_action == "c": ne_coop += 1

                if n_action == "c": n_reward = 2 * ne_coop-2
                else: n_reward = 2 * ne_coop + 1

                self.G.node[n]["reward"] = n_reward
                reward_sum += n_reward / len(neighbors)
                self.G.node[n]["agent"].update(0, n_reward) # 今状態は0だけ
                #self.G.node[n]["agent"].update(0, n_reward, n_action) # for SARSA

            # save average q value of each round
            if not self.skip_q:
                self.write_q_val(i)
            else:
                self.write_p_val(i)

            if self.share_rate is not None:
                self.update_q()
                self.share_q()

            if is_rewire:
                self.rewire_mode(rewire_lst)

            self.payoff_df[0][i] = reward_sum / self.n_agent
