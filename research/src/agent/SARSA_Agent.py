# -*- coding: utf-8 -*-

# author : Takuro Yamazaki
# description : SARSAエージェントモデル

# TODO


import numpy as np
import pandas as pd

from .policy import *
from .Agent import Agent


class SARSA_Agent(Agent):
    def __init__(self, id_: int, states: np.ndarray, actions: np.ndarray,
                 gamma: float = 0.95) -> None:
        super().__init__(id_, states, actions)

        self.__gamma = gamma
        self.n_each_action = pd.Series([0] * len(actions), index=actions)
        self.n_round = 0

        # indexが縦，columnsは横, 楽観的初期値の時はnp.onesにする
        self.q_table = pd.DataFrame(np.zeros((len(actions), len(states))), index=actions, columns=states, dtype=float)


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

        # append current reward to reward history list
        self.rewards.append(reward)

        # update q function
        self.q_table[s][a] += alpha * (reward + self.__gamma * self.q_table[state][action] - self.q_table[s][a])

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
            q_row = self.q_table[state]
            if reduction:
                action = eps_greedy(q_row, eps=max(0, 0.3 - 0.0003 * self.n_round))  # eps 減衰
            else:
                action = eps_greedy(q_row, eps=0.1)  # eps 固定

        self.prev_action = action
        self.n_each_action[action] += 1
        self.n_round += 1

        return action
