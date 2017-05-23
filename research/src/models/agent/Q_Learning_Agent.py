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

class Q_Learning_Agent:
    def __init__(self, agent_id, neighbors, state_set, action_set, gamma=0.95):
        self.agent_id = agent_id
        self.neighbors = neighbors
        self.action_set = action_set #action文字列の集合
        self.state_set = state_set #state文字列の集合
        self.gamma = gamma
        self.reward_lst = []
        self.prev_action = None
        self.action_counter = np.zeros(len(action_set))
        self.n_act = 0
        self.current_state = 0 #この初期値は妥当かな？
        #indexが縦，columnsは横, 楽観的初期値の時はnp.onesにする
        self.q_table = pd.DataFrame(np.zeros((len(action_set), len(state_set))), index=self.action_set, columns=self.state_set)
        
    def re_init(self):
        self.reward_lst = []
        self.q_table = pd.DataFrame(np.zeros((len(action_set), len(state_set))), index=self.action_set, columns=self.state_set)
        
    def get_neighbors(self):
        return self.neighbors
    
    def update_q(self, state, reward):
        a = self.action_set[self.prev_action] #今回行うアクション
        alpha = 1/(10+0.01*self.action_counter[self.prev_action])
        self.reward_lst.append(reward)
        self.q_table[self.current_state][a] += alpha*(reward+self.gamma * np.nanmax(np.array(self.q_table[state], dtype=np.float64))-self.q_table[self.current_state][a]) #q_table[state][action]
        self.current_state = state
        
    def act(self, state, random=False):
        if random:
            action = np.random.choice(np.arange(len(self.action_set)))
        else:
            q_row = self.q_table[state]
            action = eps_greedy(q_row, eps=max(0, 0.2-0.0006*self.n_act)) #eps 減衰
#             action = eps_greedy(q_row, eps=0.1) #eps 固定
        self.prev_action = action
        self.action_counter[action] += 1
        self.n_act += 1
        return self.prev_action #0 もしくは 1を返す, 0->coop, 1->comp
