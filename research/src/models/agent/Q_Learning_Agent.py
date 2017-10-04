# -*- coding: utf-8 -*-

# author : Takuro Yamazaki
# description : Q学習エージェントモデル

# TODO
#   - reward_lstの長さ制限
#   - 適格度トレースの実装(メモリ機能)
#   - alpha固定と減衰を比較

# Reference
#   - J.W.Crandall, M.A.Goodrich, "Learning to compete, compromise, and cooperate in repeated general-sum games"

import numpy as np
import pandas as pd

from .policy import *
from .Agent import Agent


class Q_Learning_Agent(Agent):
    def __init__(self, id_: int, neighbors: np.ndarray, states: np.ndarray, actions: np.ndarray, gamma: float=0.95) -> None:
        super().__init__(id_, neighbors, states, actions)

        self.__gamma = gamma
        self.n_each_action = pd.Series([0]*len(actions), index=actions)
        self.n_round = 0

        # indexが縦，columnsは横, 楽観的初期値の時はnp.onesにする
        self.q_table = pd.DataFrame(np.zeros((len(actions), len(states))), index=actions, columns=states, dtype=float)


    def update(self, state: str, reward: float) -> None:
        """
        :param state: string, state
        :param reward: float, reward value
        :return: None
        """
        a = self.prev_action
        s = self.current_state
        # alphaの設定はrefに準ずる
        alpha = 1/(10+0.01*self.n_each_action[a])

        # append current reward to reward history list
        self.reward_lst.append(reward)

        # update q function
        self.q_table[s][a] += alpha * (reward+self.__gamma*self.q_table[state].max()-self.q_table[s][a])

        self.current_state = state


    def act(self, state: str, random: bool=False, reduction: bool=False) -> str:
        """
        :param state: string, state
        :param random: boolean, random action or not
        :param reduction: boolean, reduce epsilon or not
        :return: str, selected action
        """
        if random:
            action = np.random.choice(self.__actions)
        else:
            q_row = self.q_table[state]
            if reduction:
                action_id = eps_greedy(q_row, eps=max(0, 0.2-0.0006*self.n_round)) #eps 減衰
            else:
                action_id = eps_greedy(q_row, eps=0.1) #eps 固定

            action = self.__actions[action_id]

        self.prev_action = action
        self.n_each_action[action] += 1
        self.n_round += 1
        
        return action
