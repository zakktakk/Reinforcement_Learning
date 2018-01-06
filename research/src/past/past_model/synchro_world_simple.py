#-*- coding: utf-8 -*-
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
from tqdm import tqdm

"""visualize"""
#ディスプレイ表示させない
import matplotlib as mpl
mpl.use('tkagg')

import matplotlib.pyplot as plt
plt.style.use('ggplot')

"""self made library"""
from .synchro_world import synchro_world


class synchro_world_simple(synchro_world):
    def __init__(self, n_agent, n_round, payoff_mat, nwk_alg, rl_alg,episodes=5,
                 altered_mat=None, share_rate=None, nwk_param=None, rl_param=None):
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
        super().__init__(n_agent, n_round, payoff_mat, nwk_alg, rl_alg, episodes, altered_mat, share_rate, nwk_param, rl_param,
                 None, 0)



    def run(self):
        """run simulation
        """
        rand = True
        nodes = self.G.nodes()

        for i in tqdm(range(self.n_round)):
            # 全エージェントが同期的に行動選択
            if i == len(list(self.payoff_mat))*5: rand = False

            # payoff matの更新
            if (self.altered_mat is not None) and (i == self.n_round * 0.5): self.change_payoff_metrix()

            coop_num = 0
            reward_sum = 0

            # 全てのエージェントが隣接エージェントと対戦＆更新
            for n1 in nodes:
                n2 = np.random.choice(self.G.neighbors(n1))
                
                n1_action = self.G.node[n1]["agent"].act(0, random=rand)
                n2_action = self.G.node[n2]["agent"].act(0, random=rand)

                if n1_action == "c": coop_num += 1

                n1_reward = self.payoff_mat[n2_action][n1_action]
                n2_reward = self.payoff_mat[n1_action][n2_action]

                reward_sum += n1_reward
                self.G.node[n1]["reward"] = n1_reward

                self.G.node[n1]["agent"].update(0, n1_reward) # 今状態は0だけ
                #self.G.node[n1]["agent"].update(0, n1_reward, n1_action) # for SARSA

                self.G.node[n2]["agent"].update(0, n2_reward)
                #self.G.node[n2]["agent"].update(0, n2_reward, n2_action) # for SARSA

            # save average q value of each round
            if not self.skip_q:
                self.write_q_val(i)
            else:
                self.write_p_val(i)

            if self.share_rate is not None:
                self.update_q()
                self.share_q()

            self.coop_per_df[0][i] = coop_num / self.n_agent
            self.payoff_df[0][i] = reward_sum / self.n_agent