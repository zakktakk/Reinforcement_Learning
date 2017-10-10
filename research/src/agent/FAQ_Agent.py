#-*- coding: utf-8 -*-

# author : Takuro Yamazaki
# description : Lenient frequency adjusted Q-learning エージェントモデル

# Reference
#   - D. Bloembergen, M. Kaisers, K. Tuyls, "Lenient Frequency Adjusted Q-Learning", In Belgian/Netherlands Artificial Intelligence Conference, 2010
# これはマルチエージェント用の手法だからこれやっても意味ない木がするぞ


import numpy as np
import pandas as pd
from .Agent import Agent
from .policy import eps_greedy, softmax_boltzman



class LFAQ_Agent(Agent):
    def __init__(self, id_: int, states: np.ndarray, actions: np.ndarray,
                 gamma: float=0.95, beta: float=10**(-3)):

        super().__init__(id_, states, actions)
        self.gamma = gamma
        self.n_each_action = pd.Series([0]*len(actions), index=actions)
        self.n_round = 0
        self.__beta = beta

        #indexが縦，columnsは横, 楽観的初期値の時はnp.onesにする
        self.q_table = pd.DataFrame(np.zeros((len(actions), len(states))), index=actions, columns=states)


    def update(self, state: str, reward: float) -> None:
        a = self.prev_action
        s = self.current_state

        alpha = 1/(10+0.01*self.n_each_action[self.prev_action]) #先行研究に準じたalpha
        self.rewards.append(reward)

        xi = softmax_boltzman(self.q_table[state])[self.actions.index(a)] #action選択される確率
        fa_val = np.min(1, self.__beta/xi)

        #q tableの更新
        self.q_table[s][a] += fa_val*alpha*(reward+self.gamma*self.q_table[state].max()-self.q_table[s][a])

        #状態の更新
        self.current_state = state


    def act(self, state: str, random: bool=False, reduction: bool=True) -> str:
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
                action_id = eps_greedy(q_row, eps=max(0, 0.2-0.0006*self.n_round)) #eps 減衰
            else:
                action_id = eps_greedy(q_row, eps=0.1) #eps 固定

            action = self.actions[action_id]

        self.prev_action = action
        self.n_each_action[action] += 1
        self.n_round += 1

        return action
