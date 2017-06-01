#-*- coding: utf-8 -*-

# author : Takuro Yamazaki
# last update : 05/23/2017
# description : Q学習エージェントモデル

# TODO
#   - reward_lstの長さ制限
#   - 適格度トレースの実装(メモリ機能)

# Reference
#   - J.W.Crandall, M.A.Goodrich, "Learning to compete, compromise, and cooperate in repeated general-sum games", In ICML, 2005

import pandas as pd
import numpy as np
import sys
from agent.policy import eps_greedy
from agent.Agent import Agent

class Q_Learning_Agent(Agent):
    def __init__(self, agent_id, neighbors, state_set, action_set, gamma=0.95):
        super().__init__(agent_id, neighbors, state_set, action_set)
        self.gamma = gamma
        self.n_strategy = np.zeros(self.len_a)
        self.n_act = 0

        #indexが縦，columnsは横, 楽観的初期値の時はnp.onesにする
        self.q_tb = pd.DataFrame(np.zeros((self.len_a, self.len_s)), index=self.action_set, columns=self.state_set)

    def update_q(self, state, reward):
        a = self.action_set[self.prev_action] #今回行うアクション
        alpha = 1/(10+0.01*self.n_strategy[self.prev_action])
        self.reward_lst.append(reward)
        self.q_tb[self.c_st][a] += alpha*(reward+self.gamma*np.nanmax(np.array(self.q_tb[state], dtype=np.float64))-self.q_tb[self.c_st][a]) #q_tb[state][action]
        self.c_st = state

    def act(self, state, random=False, reduction=False):
        if random:
            action = np.random.choice(np.arange(self.len_a))
        else:
            q_row = self.q_tb[state]
            if reduction:
                action = eps_greedy(q_row, eps=max(0, 0.2-0.0006*self.n_act)) #eps 減衰
            else:
                action = eps_greedy(q_row, eps=0.1) #eps 固定

        self.prev_action = action
        self.n_strategy[action] += 1
        self.n_act += 1

        return action #0 もしくは 1を返す, 0->coop, 1->comp
