# author : Takuro Yamazaki
# last update : 2017/09/02

import numpy as np
from agent import Agent

class Game:
    def __init__(self, n_agents, n_strategies):
        self.n_agents = n_agents
        self.n_strategies = n_strategies # joint strategyだから違うかも


    def __generate_agents(self, n_agents, n_strategies):
        self.agent_lst = []
        for i in range(n_agents):
            self.agent_lst.append(Agent(i, n_strategies))
