#-*- coding: utf-8 -*-

# author : Takuro Yamazaki
# last update : 05/23/2017
# description : エージェントの親モデル

# TODO
# 引数の調整

class Agent(object):
    def __init__(self, agent_id, neighbors, state_set, action_set):
        self.agent_id = agent_id
        self.neighbors = neighbors
        self.action_set = action_set #action文字列の集合
        self.state_set = state_set #state文字列の集合
        self.reward_lst = []
        self.prev_action = None
        self.current_state = 0

    def re_init(self):
        pass

    def get_neighbors(self):
        return self.neighbors
    
    def update_q(self, state, reward):
        pass

    def act(self, state, random=False):
        pass
