#-*- coding: utf-8 -*-

# author : Takuro Yamazaki
# last update : 05/27/2017
# description : M-Qubedエージェントモデル

# Reference
# 1個目と2個目で定義が違うから気をつけて！
#   - J.W.Crandall, M.A.Goodrich, "Learning to compete, compromise, and cooperate in repeated general-sum games", In ICML, 2005
#   - J.W.Crandall, M.A.Goodrich, "Learning to compete, coordinate, and cooperate in repeated games using reinforcement learning", Mach Learn, 82, 2011

#history parameter ωをパラメータとして導入する

import pandas as pd
import numpy as np
import sys
from agent.policy import eps_greedy
from agent.Agent import Agent

class MQubed_Agent(Agent):
    def __init__(self, agent_id, neighbors, state_set, action_set, payoff_matrix, gamma=0.95):
        super().__init__(agent_id, neighbors, state_set, action_set)

        self.n_act = 0 #actionを何回行ったか
        self.h = np.nan #初手nan
        self.l = np.nan
        self.beta = np.zeros(len(action_set))
        self.gamma = gamma
        self.xi = 0.05+np.random.rand()*0.05 #0.05~0.1の乱数

        self.payoff_matrix = payoff_matrix
        self.joint_mm_val = self.minimax_val()*len(neighbors)

        #joint actionを隣接中のcoopの数とする, min 0, max len(neighbots)
        #TODO action数が3以上だとできないから汎用性考える
        self.joint_action_set = np.arange(len(neighbors))
        #pi tableの長さはjoint actionの数
        self.pi_tb = pd.DataFrame(np.ones((self.len_a, len(joint_action_set)))/len(self.len_a), index=self.action_set, columns=self.joint_action_set)
        self.pi_br_tb = pd.DataFrame(np.zeros((self.len_a, len(joint_action_set))), index=self.action_set, columns=self.joint_action_set)
        self.pi_cs_tb = pd.DataFrame(np.zeros((self.len_a, len(joint_action_set))), index=self.action_set, columns=self.joint_action_set)

        #q table
        self.q_tb = pd.DataFrame(np.ones((self.len_a, len(joint_action_set)))/(1-gamma), index=self.action_set, columns=self.joint_action_set)

        #実際L_tolはそのタイムステップで可能なアクション数をかけるから時間に依存するが
        #今回そんな自体は起こらないので定数として扱う
        self.L_tol = 500*len(self.joint_action_set)*self.len_a

    def act(self, state):
        #確率piで行動選択
        return np.random.choice(np.arange(self.len_a), p=np.array(self.pi_tb[state]))

    def minimax_val(self):
        #own payoff matrix
        own_payoff = self.payoff_matrix[:,:,0]
        mm_val = np.max(np.min(own_payoff, axis=1))
        return mm_val

    def minimax_act(self):
        #own payoff matrix
        own_payoff = self.payoff_matrix[:,:,0]
        action = np.argmax(np.min(own_payoff, axis=1))
        return action

    def update_q(self, state, reward):
        """
        @param state いつもは状態1だけだけど今はjoint actionにする
        """
        a = self.action_set[self.prev_action] #今回行うアクション
        num = (self.xi/(1-self.joint_mm_val+self.xi)) ** (self.xi*self.len_a*len(self.joint_action_set)/self.L_tol)
        alpha = (1-num)/(1-self.gamma)
        self.reward_lst.append(reward)

        #L accumの計算
        self.L_accum = max(0, self.joint_mm_val-np.mean(self.reward_lst))

        #betaの設定
        if self.L_tol <= self.L_accum:
            self.beta = 1
        else:
            self.beta = (self.L_accum/self.L_tol)**10

        #############333この2パートは統合できる
        #update pi upper
        if np.max(self.q_tb[state])>self.joint_mm_val/(1-self.gamma):
            swp = np.array([0,0])
            swp[np.argmax(self.q_tb[state])] = 1
            self.pi_br_tb[state] = swp
        else:
            swp = np.array([0,0])
            swp[self.minimax_act()] = 1
            self.pi_br_tb[state] = swp

        #update pi lower
        if L_accum < L_tol:
            swp = np.array([0,0])
            swp[np.argmax(self.q_tb[state])] = 1
            self.pi_cs_tb[state] = swp
        else:
            swp = np.array([0,0])
            swp[self.minimax_act()] = 1
            self.pi_br_tb[state] = swp
        #############333この2パートは統合できる

        self.pi_star[state] = self.beta*self.pi_br_tb[state] + (1-self.beta)*self.pi_cs_tb[state]

        #S_star : max Q値に近いQちをもつSの集合
        max_q = self.q_tb.max()
        self.q_tb.max(axis=0)


        #pi tableの更新
        if self.beta == 1||
        self.pi_tb[state]

        #calc v
        v = np.sum(self.pi_tb[state]*self.q_tb[state])
        #update q table
        self.q_tb[self.c_st][a] += alpha*(reward+self.gamma*v-self.q_tb[self.c_st][a])

        self.c_st = state
