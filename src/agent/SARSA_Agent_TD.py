# -*- coding: utf-8 -*-

# author : Takuro Yamazaki
# description : SARSA td(lmd)エージェントモデル


import numpy as np
import pandas as pd

from .policy import *
from .Agent import Agent


class SARSA_Agent_TD(Agent):
    def __init__(self, id_: int, states: np.ndarray, actions: np.ndarray, gamma: float=0.95,
                 lmd=0.5) -> None:
        super().__init__(id_, states, actions)

        self.__gamma = gamma
        self.__lmd = lmd
        self.n_each_action = pd.Series([0] * len(actions), index=actions)
        self.n_round = 0

        # indexが縦，columnsは横, 楽観的初期値の時はnp.onesにする
        self.q_df = pd.DataFrame(np.zeros((len(actions), len(states))), index=actions, columns=states, dtype=float)

        # 適格度トレースのテーブル
        self.e_df = pd.DataFrame(np.zeros((len(actions), len(states))), index=actions, columns=states, dtype=float)



    def update(self, state: str, reward: float, action: str) -> None:
        """
        :param state: string, state
        :param reward: float, reward value
        :return: None
        """
        a = self.prev_action
        s = self.current_state
        # alphaの設定はrefに準ずる
        alpha = 1 / (10 + 0.01 * self.n_each_action[a])

        delta = reward + self.__gamma * self.q_df[state][action] - self.q_df[s][a]

        # update eligibility trace df
        self.e_df[s][a] += 1

        # update all q_df and e_df
        for ua in self.actions:
            for us in self.states:
                self.q_df[us][ua] += alpha * delta * self.e_df[us][ua]
                self.e_df[us][ua] *= self.__gamma * self.__lmd

        self.current_state = state


    def act(self, state: str, random: bool = False, reduction: bool = True) -> str:
        """
        :param state: string, state
        :param random: boolean, random action or not
        :param reduction: boolean, reduce epsilon or not
        :return: str, selected action
        """
        if random:
            action = np.random.choice(self.actions)
        else:
            q_row = self.q_df[state]
            if reduction:
                action = eps_greedy(q_row, eps=max(0, 0.2-0.0002*self.n_round))  # eps 減衰
            else:
                action = eps_greedy(q_row, eps=0.1)  # eps 固定

        self.prev_action = action
        self.n_each_action[action] += 1
        self.n_round += 1

        return action
