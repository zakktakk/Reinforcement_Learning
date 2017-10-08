#-*- coding: utf-8 -*-

# author : Takuro Yamazaki
# description : Satisficing algorithmエージェントモデル

# TODO

# Reference
# http://www.aaai.org/Papers/ICML/2003/ICML03-095.pdf
# ↑のやつ下で参照してるのとは結構違うんだけど下信頼していいのだろうか
# https://kaigi.org/jsai/webprogram/2014/pdf/654.pdf

import random
import numpy as np

from .Agent import Agent

class Satisficing_Agent(Agent):
    def __init__(self, id_: int, states: np.ndarray, actions: np.ndarray, R_max:float, lmd:float=0.95):
        self.lmd =lmd
        super().__init__(id_, states, actions)
        self.asp = random.uniform(R_max, 2*R_max)


    def update(self, reward:float) -> None:
        """
        :param reward: float, reward value
        :return: None
        """
        self.reward_lst.append(reward)
        self.asp = self.asp * self.lmd + (1-self.lmd) * reward


    def act(self, state: str, random: bool=False) -> str:
        """
        :param state: string, state
        :param random: boolean, random action or not
        :return: str, selected action
        """
        if random or self.reward_lst[-1] < self.asp:
            # ランダムか閾値を超えない
            action = np.random.choice(self.__actions)
        else:
            action = self.prev_action

        self.prev_action = action
        return action
