#-*- coding: utf-8 -*-

# author : Takuro Yamazaki
# description : WoLF-PHCエージェントモデル

# TODO

# Reference
# http://studentnet.cs.manchester.ac.uk/resources/library/3rd-year-projects/2015/yifei.wang-6.pdf, p14

import pandas as pd
import numpy as np
from .Agent import Agent

class WoLF_PHC_Agent(Agent):
    def __init__(self, id_: int,  states: np.ndarray, actions: np.ndarray, gamma:float=0.9):
        super().__init__(id_, states, actions)
        len_s = len(states)
        len_a = len(actions)
        self.__gamma = gamma
        #indexが縦，columnsは横, 楽観的初期値の時はnp.onesにする -> これ実装する？
        self.q_table = pd.DataFrame(np.zeros((len_a, len_s)),index=actions, columns=states, dtype=np.float64)

        self.count_table = self.q_table.copy()
        self.pi_d_table = self.q_table.copy()

        self.pi_table = pd.DataFrame(np.ones((len_a, len_s))/len_a,
                                     index=actions, columns=states)

        self.C = pd.DataFrame(np.zeros(len_s), columns=states)


    def __q_mean(self, pi_table: pd.DataFrame, q_table: pd.DataFrame) -> float:
        """
        :param pi_table: pandas dataframe, pi value table
        :param q_table: pandas dataframe, q value table
        :return: pi * q summuation value
        """
        return np.sum(np.array(pi_table[self.current_state])*np.array(q_table[self.current_state]))


    def update(self, state: str, reward: float) -> None:
        """

        :param state: string, state
        :param reward: float, reward value
        :return: None
        """
        a = self.prev_action
        s = self.current_state

        alpha = 1/(10+0.01*self.count_table[s][a])
        delta_w = 1/(10+self.count_table[s][a])
        delta_l = 4*delta_w
        self.reward_lst.append(reward)

        # update q table
        self.q_table[s][a] += alpha/(self.C[s]+1)*(reward+self.__gamma*self.q_table[state].max()-self.q_table[s][a])


        # update estimate of average policy pi dash
        self.C[s] += 1
        self.pi_d_table[s] += (self.pi_table[s]-self.pi_d_table[s])/self.C[s][0]


        # update pi and constrain it to legal probability distribution
        if self.__q_mean(self.pi_table, self.q_table) > self.__q_mean(self.pi_d_table, self.q_table):
            delta = delta_w / self.C[s][0]
        else:
            delta = delta_l / self.C[s][0]


        if self.prev_action == self.__actions[self.q_table[s].argmax()]: # 怪しい
            self.pi_table[s][a] += delta
        else:
            self.pi_table[s][a] -= delta / (len(self.__actions)-1)


        self.pi_table[s] /= self.pi_table[s].sum()

        # update current state
        self.current_state = state



    def act(self, state:str, random=True) -> str:
        """
        :param state: string, state
        :param random: boolean, random action or not
        :return: str, selected action
        """
        action = np.random.choice(self.__actions, p=np.array(self.pi_table[state]))

        self.prev_action = action
        self.count_table[state][action] += 1

        return action
