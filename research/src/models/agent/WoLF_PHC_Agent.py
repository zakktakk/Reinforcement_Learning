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
        self.q_table = pd.DataFrame(np.zeros((len(action_set), len(state_set))),
                                    index=self.action_set, columns=self.state_set)

        self.pi_table = pd.DataFrame(np.ones((len(action_set), len(state_set)))/len(action_set),
                                     index=self.action_set, columns=self.state_set)
        
        self.count_table = pd.DataFrame(np.zeros((len(action_set), len(state_set))),
                                        index=self.action_set, columns=self.state_set)
        
        self.C = pd.DataFrame(np.zeros(len(state_set)), columns=self.state_set)
        
        self.pi_d_table = pd.DataFrame(np.zeros((len(action_set), len(state_set))),
                                       index=self.action_set, columns=self.state_set)
        
    
    def re_init(self):
        self.reward_lst = []
        self.q_table = pd.DataFrame(np.zeros((len(action_set), len(state_set))),
                                    index=self.action_set, columns=self.state_set)


    def q_mean(self, pi_table, q_table):
        return np.sum(np.array(pi_table[self.current_state])*np.array(q_table[self.current_state]))
    

    def update_q(self, state, reward):
        a = self.action_set[self.prev_action] #今回行うアクション
        alpha = 1/(10+0.01*self.count_table[self.current_state][a])
        delta_w = 1/(10+self.count_table[self.current_state][a])
        delta_l = 4*delta_w
        self.reward_lst.append(reward)
        
        #update q table
        self.q_table[self.current_state][a] += alpha/(self.C[self.current_state]+1)*(reward+self.gamma*np.nanmax(np.array(self.q_table[state], dtype=np.float64))-self.q_table[self.current_state][a]) #q_table[state][action]
        
        #update estimate of average policy pi dash
        self.C[self.current_state] += 1
        self.pi_d_table[self.current_state] += (self.pi_table[self.current_state]-self.pi_d_table[self.current_state])/self.C[self.current_state][0]

        #update pi and constrain it to legal probability distribution
        if self.q_mean(self.pi_table, self.q_table) > self.q_mean(self.pi_d_table, self.q_table):
            delta = delta_w / self.C[self.current_state][0]
        else:
            delta = delta_l / self.C[self.current_state][0]
        
        if self.prev_action == np.nanargmax(np.array(self.q_table[self.current_state])):
            self.pi_table[self.current_state][a] += delta
        else:
            self.pi_table[self.current_state][a] -= delta / (len(self.action_set)-1)
        
        self.pi_table[self.current_state] /= np.nansum(np.array(self.pi_table[self.current_state]))
        self.current_state = state
        

    def act(self, state, random=False):
        if random:
            action = 0
        else:
            action = np.random.choice(np.arange(len(self.action_set)), p=np.array(self.pi_table[state]))

        self.prev_action = action
        self.count_table[state][self.action_set[action]] += 1
        return self.prev_action #0 もしくは 1を返す, 0->coop, 1->comp
