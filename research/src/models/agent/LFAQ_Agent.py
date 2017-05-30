#-*- coding: utf-8 -*-

# author : Takuro Yamazaki
# last update : 05/30/2017
# description : Lenient frequency adjusted Q-learning エージェントモデル

# Reference
#   - D. Bloembergen, M. Kaisers, K. Tuyls, "Lenient Frequency Adjusted Q-Learning", In Belgian/Netherlands Artificial Intelligence Conference, 2010

import pandas as pd
import numpy as np
import sys
from agent.policy import eps_greedy, softmax_boltzman
from agent.Agent import Agent

class LFAQ_Agent(Agent):
    def __init__(self, agent_id, neighbors, state_set, action_set, gamma=0.95, beta=0.1):
        super().__init__(agent_id, neighbors, state_set, action_set)
        self.gamma = gamma
        self.action_counter = np.zeros(self.len_a)
        self.n_act = 0
        self.beta = beta
        
        #indexが縦，columnsは横, 楽観的初期値の時はnp.onesにする
        self.q_table = pd.DataFrame(np.zeros((self.len_a, self.len_s)), index=self.action_set, columns=self.state_set)
        

    def re_init(self):
        self.reward_lst = []
        self.q_table = pd.DataFrame(np.zeros((self.len_a, self.len_s)), index=self.action_set, columns=self.state_set)
        

    def update_q(self, state, reward):
        a = self.action_set[self.prev_action]
        alpha = 1/(10+0.01*self.action_counter[self.prev_action]) #先行研究に準じたalpha
        self.reward_lst.append(reward)
        
        xi = softmax_boltzman(self.q_table[state])[self.prev_action]
        fa_val = np.min(1, self.beta/xi)

        #q tableの更新
        self.q_table[self.c_state][a] += fa_val*alpha*(reward+self.gamma*np.nanman(np.array(self.q_table[state], dtype=np.float64))-self.q_table[self.c_state][a])

        #状態の更新
        self.c_state = state


    def act(self, state, random=False, reduction=False):
        if random:
            action = np.random.choice(np.arange(self.len_a))
        else:
            q_row = self.q_table[state]
            if reduction:
                action = eps_greedy(q_row, eps=max(0, 0.2-0.0006*self.n_act)) #eps 減衰
            else:
                action = eps_greedy(q_row, eps=0.1) #eps 固定
        
        self.prev_action = action
        self.action_counter[action] += 1
        self.n_act += 1

        return self.prev_action #0 もしくは 1を返す, 0->coop, 1->comp
