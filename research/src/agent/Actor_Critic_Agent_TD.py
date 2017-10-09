#-*- coding: utf-8 -*-

# author : Takuro Yamazaki
# description : Actor Criticのエージェントモデル, actorはsoftmax bortzman, criticはTD(λ)学習
# TODO 全然まだ

# ref : http://stlab.ssi.ist.hokudai.ac.jp/yuhyama/lecture/OLD/softcomputing/softcomputing-b-4up.pdf

import numpy as np
import pandas as pd

from .policy import *
from .Agent import Agent


class Actor_Critic_Agent_TD(Agent):
    def __init__(self, id_: int, states: np.ndarray, actions: np.ndarray, beta: float=1,
                 lmd: float=0.5, gamma: float=0.95) -> None:
        """
        :param beta: 正のステップサイズ変数
        """
        super().__init__(id_, states, actions)

        self.__beta = beta
        self.__gamma = gamma
        self.__lmd = lmd
        self.n_each_action = pd.Series([0] * len(actions), index=actions)
        self.n_round = 0
        self.T = 1 # 温度パラメータ

        self.v_table = pd.Series(np.zeros(len(states)), index=states, dtype=float)
        self.p_table = pd.DataFrame(np.zeros((len(actions), len(states))), index=actions, columns=states, dtype=float)

        # 適格度トレースのテーブル
        self.e_table = pd.Series(np.zeros(len(states)), index=states, dtype=float)


    def update(self, state: str, reward: float) -> None:
        """
        :param state: string, state
        :param reward: float, reward value
        :return: None
        """
        a = self.prev_action
        s = self.current_state

        alpha = 1 / (10 + 0.01 * self.n_each_action[a])

        # append current reward to reward history list
        self.rewards.append(reward)

        # update v table
        delta = reward + self.__gamma * self.v_table[state] - self.v[s]

        # update eligibility trace table
        self.e_table[s][a] += 1

        # update all v_table and e_table
        for us in self.__states:
            self.v_table[us] += alpha * delta + self.e_table[us]
            self.e_table[us] *= self.__gamma * self.__lmd

        # update p table
        self.p_table[s][a] += delta * self.__beta
        # update current state
        self.current_state = state


    def act(self, state: str, random: bool = False) -> str:
        """
        :param state: string, state
        :param random: boolean, random action or not
        :param reduction: boolean, reduce epsilon or not
        :return: str, selected action
        """
        if random:
            action = np.random.choice(self.__actions)
        else:
            q_row = self.p_table[state]
            action_id = softmax_boltzman(q_row, T=self.T)
            action = self.__actions[action_id]
            self.T *= 0.9

        self.prev_action = action
        self.n_each_action[action] += 1
        self.n_round += 1

        return action
