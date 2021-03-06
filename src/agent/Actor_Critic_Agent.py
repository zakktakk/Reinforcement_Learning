#-*- coding: utf-8 -*-

# author : Takuro Yamazaki
# description : Actor Criticのエージェントモデル, actorはsoftmax bortzman, criticはTD(0)学習

# ref : http://stlab.ssi.ist.hokudai.ac.jp/yuhyama/lecture/OLD/softcomputing/softcomputing-b-4up.pdf

import numpy as np
import pandas as pd

from .policy import *
from .Agent import Agent


class Actor_Critic_Agent(Agent):
    def __init__(self, id_: int, states: np.ndarray, actions: np.ndarray, reguralize_value:int, gamma: float=0.95,
                 alpha: float=None, beta: float=1) -> None:
        """
        :param beta: 正のステップサイズ変数
        """
        super().__init__(id_, states, actions)

        self.__beta = beta
        self.__gamma = gamma
        self.n_each_action = pd.Series([0] * len(actions), index=actions)
        self.prev_prob = 0.5
        self.n_round = 0
        self.regurelize_value = reguralize_value
        self.T = 1 # 温度パラメータ
        self.alpha = alpha

        self.v_df = pd.Series(np.zeros(len(states)), index=states, dtype=float)
        self.p_df = pd.DataFrame(np.zeros((len(actions), len(states))), index=actions, columns=states, dtype=float)


    def update(self, state: str, reward: float) -> None:
        """
        :param state: string, state
        :param reward: float, reward value
        :return: None
        """
        a = self.prev_action
        s = self.current_state
        reward /= self.regurelize_value

        if self.alpha is None:
            alpha = 1 / (10 + 0.01 * self.n_each_action[a])
        else:
            alpha = self.alpha

        # update v df
        delta = reward + self.__gamma * self.v_df[state] - self.v_df[s]
        self.v_df[s] += alpha * delta

        self.p_df[s][a] += delta * self.__beta * (1-self.prev_prob)
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
            action = np.random.choice(self.actions)
        else:
            q_row = self.p_df[state]
            action_id, self.prev_prob = softmax_boltzman(q_row, T=self.T)
            action = self.actions[action_id]
            self.T *= 0.999
            self.T = max(0.1, self.T)

        self.prev_action = action
        self.n_each_action[action] += 1
        self.n_round += 1

        return action
