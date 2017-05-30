#-*- coding: utf-8 -*-

# author : Takuro Yamazaki
# last update : 05/27/2017
# description : M-Qubedエージェントモデル

# Reference
#   - J.W.Crandall, M.A.Goodrich, "Learning to compete, compromise, and cooperate in repeated general-sum games", In ICML, 2005

import pandas as pd
import numpy as np
import sys
from agent.policy import eps_greedy
from agent.Agent import Agent

class MQubed_Agent(Agent):
    def __init__(self, agent_id, neighbors, state_set, action_set,  n_init_act, alpha=0.1, gamma=0.95):
        super().__init__(agent_id, neighbors, state_set, action_set)
        self.alpha = alpha
        self.gamma = gamma
        self.n_init_act = n_init_act #初期行動を何回行うか
        self.n_act = 0 #actionを何回行ったか

        #行動価値関数
        self.q_table = pd.DataFrame(np.zeros((self.len_a, self.len_s)), index=self.action_set, columns=self.state_set)

        #価値関数
        self.v_table = pd.DataFrame(np.zeros(self.len_s), columns=self.state_set)

        #方策
        self.pi_table = pd.DataFrame(np.zeros((self.len_a, self.len_s)), index=self.action_set, columns=self.state_set)

        #minimax方策のための報酬テーブル
        self.mini_r_table = pd.DataFrame(np.zeros((self.len_a, self.len_s))), 

    def re_init(self):
        self.n_act = 0
        self.reward_lst = []
        self.q_table = pd.DataFrame(np.zeros((self.len_a, self.len_s)), index=self.action_set, columns=self.state_set)

    def init_act(self):
        #これ相手の戦略は自分と同じという仮定のもとで行うのか？
        """
        @description 初期行動を表す. ランダムに行動し，報酬のminmaxを推定
        @return ランダム行動
        """
        #minimaxテーブルの学習を行う -> TODO これはinit_actが終了しても行うからactに入れてしまっていいのでは?
        return np.random.choice(np.arange(self.len_a))

    def minimax_act(self):
        # 現在の状態に基づいてminimax戦略を返す
        return np.nanargmax(np.array(self.mini_r_table)[self.c_state])
    
    def update_q(self, state, reward):
        a = self.action_set[self.prev_action]
        self.reward_lst.append(reward)

        #update q table
        self.q_table[self.c_state][a] += alpha*(reward+self.gamma*np.nanmax(np.array(self.q_table[state], dtype=np.float64))-self.q_table[self.c_state][a]) #q_table[state][action]

        self.c_state = state

        self.v_table
