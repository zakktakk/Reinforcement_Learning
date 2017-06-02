#-*- coding: utf-8 -*-

# author : Takuro Yamazaki
# last update : 05/23/2017
# description : WoLF-PHCエージェントモデル

# TODO

# Reference
# http://studentnet.cs.manchester.ac.uk/resources/library/3rd-year-projects/2015/yifei.wang-6.pdf, p14

import pandas as pd
import numpy as np
from agent.Agent import Agent

class WoLF_PHC_Agent(Agent):
    def __init__(self, agent_id, neighbors, state_set, action_set, gamma=0.9):
        super().__init__(agent_id, neighbors, state_set, action_set)
        self.gamma = gamma
        #indexが縦，columnsは横, 楽観的初期値の時はnp.onesにする
        self.q_tb = pd.DataFrame(np.zeros((self.len_a, self.len_s)),
                                    index=self.action_set, columns=self.state_set)

        self.pi_tb = pd.DataFrame(np.ones((self.len_a, self.len_s))/self.len_a,
                                     index=self.action_set, columns=self.state_set)

        self.count_tb = pd.DataFrame(np.zeros((self.len_a, self.len_s)),
                                        index=self.action_set, columns=self.state_set)

        self.C = pd.DataFrame(np.zeros(self.len_s), columns=self.state_set)

        self.pi_d_tb = pd.DataFrame(np.zeros((self.len_a, self.len_s)),
                                       index=self.action_set, columns=self.state_set)

    def q_mean(self, pi_tb, q_tb):
        return np.sum(np.array(pi_tb[self.c_st])*np.array(q_tb[self.c_st]))


    def update_q(self, state, reward):
        a = self.action_set[self.prev_action] #今回行うアクション
        alpha = 1/(10+0.01*self.count_tb[self.c_st][a])
        delta_w = 1/(10+self.count_tb[self.c_st][a])
        delta_l = 4*delta_w
        self.reward_lst.append(reward)

        #update q tb
        self.q_tb[self.c_st][a] += alpha/(self.C[self.c_st]+1)*(reward+self.gamma*np.nanmax(np.array(self.q_tb[state], dtype=np.float64))-self.q_tb[self.c_st][a]) #q_tb[state][action]

        #update estimate of average policy pi dash
        self.C[self.c_st] += 1
        self.pi_d_tb[self.c_st] += (self.pi_tb[self.c_st]-self.pi_d_tb[self.c_st])/self.C[self.c_st][0]

        #update pi and constrain it to legal probability distribution
        if self.q_mean(self.pi_tb, self.q_tb) > self.q_mean(self.pi_d_tb, self.q_tb):
            delta = delta_w / self.C[self.c_st][0]
        else:
            delta = delta_l / self.C[self.c_st][0]

        if self.prev_action == np.nanargmax(np.array(self.q_tb[self.c_st])):
            self.pi_tb[self.c_st][a] += delta
        else:
            self.pi_tb[self.c_st][a] -= delta / (len(self.action_set)-1)

        self.pi_tb[self.c_st] /= np.nansum(np.array(self.pi_tb[self.c_st]))

        #update current state
        self.c_st = state


    def act(self, state):
        action = np.random.choice(np.arange(len(self.action_set)), p=np.array(self.pi_tb[state]))

        self.prev_action = action
        self.count_tb[state][self.action_set[action]] += 1
        return action #0 もしくは 1を返す, 0->coop, 1->comp
