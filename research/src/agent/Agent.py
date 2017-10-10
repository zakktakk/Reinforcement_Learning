#-*- coding: utf-8 -*-

# author : Takuro Yamazaki
# description : エージェントの親モデル

import numpy as np
from abc import abstractmethod, ABCMeta


class Agent(metaclass=ABCMeta):
    def __init__(self, id_: int, states: np.ndarray, actions: np.ndarray):

        """superclass of Agent
        :param id_ : int,
            エージェントのID

        :param neighbors : array, shape=[n_neighbors]
            隣接するエージェントのarray

        :param states : array, shape=[n_states]
            状態を表す文字列の集合

        :param actions : array, shape=[n_actions]
            行動を表す文字列の集合
            順序が欲しくかつdiffなどの操作がしやすいためndarrayにした

        :param rewards : list, len=n_history
            報酬の履歴を表す, append操作のためにlistにした

        :param prev_action : sting,
            一つ前の行動のstring

        :param current_state : string,
            現在の状態のstring
        """

        self.id_ = id_

        self.actions = actions
        self.states = states
        self.rewards = []

        self.prev_action = ""
        self.current_state = ""


    @abstractmethod
    def update(self, *args, **kwargs):
        """内部状態(Q関数など)の更新
        """
        pass

    @abstractmethod
    def act(self, *args, **kwargs):
        """内部状態を元に行動を決定
        """
        pass
