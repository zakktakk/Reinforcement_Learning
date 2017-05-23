#-*- coding: utf-8 -*-

# author : Takuro Yamazaki
# last update : 05/23/2017
# description : Satisficing algorithmエージェントモデル

# TODO

# Reference
# http://www.aaai.org/Papers/ICML/2003/ICML03-095.pdf
# ↑のやつ下で参照してるのとは結構違うんだけど下信頼していいのだろうか
# https://kaigi.org/jsai/webprogram/2014/pdf/654.pdf

import panda as pd
import numpy as np
import Agent.Agent

class Satisficing_Agent(Agent.Agent):
    def __init__(self, agent_id, neighbors, state_set, action_set,R_max, lmd=0.95):
        self.lmd =lmd
        super().__init__(agent_id, neighbors, state_set, action_set)
        self.R_max = R_max
        self.asp = random.uniform(R_max, 2*R_max)
        
    def re_init(self):
        self.reward_lst = []
        self.total_r = 0
        self.asp = random.uniform(R_max, 2*R_max)
        
    def update_q(self, reward): #便宜上_qにしてるけど変えたほうがいい
        self.reward_lst.append(reward)
        self.asp = self.asp * self.lmd + (1-self.lmd) * reward
            
    def act(self, state, random=False):
        if random or self.reward_lst[-1]<self.asp:
            action = np.random.choice(np.arange(len(self.action_set)))
        else:
            action = self.prev_action
        self.prev_action = action
        return self.prev_action #0 もしくは 1を返す, 0->coop, 1->comp
