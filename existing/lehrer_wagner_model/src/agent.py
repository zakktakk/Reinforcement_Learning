# author : Takur Yamazaki
# last update : 2017/09/02

import numpy as np
from scipy.stats import uniform

class Agent:
    def __init__(self, agent_id, n_strategies):
        self.agent_id = agent_id
        self.n_strategies = n_strategies
        self.u = np.random.rand()
        self.b = uniform

