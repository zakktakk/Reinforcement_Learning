#-*- coding: utf-8 -*-
# author : Takuro Yamazaki
# last update : 05/23/2017
# description : 同時プレイの世界

class synchro_world:
    def __init__(self, n_agent, n_round, payoff_matrix):
        self.n_agent = n_agent
        self.n_round = n_round
        self.action_name, self.payoff_matrix = payoff_matrix
        self.create_network(n_agent)
        self.agent_action_table = pd.DataFrame(np.zeros((n_round, n_agent)))
        self.agent_payoff_table = pd.DataFrame(np.zeros((n_round, n_agent)))
        
    def create_network(self, n_coop_agent):
        self.G = nx.barabasi_albert_graph(n_coop_agent, 30)
        print(len(self.G.edges()))
        self.n_edges = 0
        for n in self.G.nodes():
            self.n_edges += len(self.G.neighbors(n))
            #agent_id, neighbors, state_set, action_set,R_max, lmd=0.95
            agent = Q_Learning_Agent(n, sorted(self.G.neighbors(n)), [0], self.action_name)
            self.G.node[n]["agent"] = agent
            self.G.node[n]["action"] = 0
            
    def run(self):
        rand = True
        for i in range(self.n_round):
            #全エージェントが同期的に行動選択
            if i == len(self.payoff_matrix)*5:
                rand=False
            for n in  self.G.nodes():
                self.G.node[n]["action"] = self.G.node[n]["agent"].act(0, rand)
                self.agent_action_table[n][i] = self.G.node[n]["action"]
            #報酬計算&Q値更新
            for n in self.G.nodes():
                neighbors = self.G.node[n]["agent"].get_neighbors()
                n_action = self.G.node[n]["action"]
                n_reward = 0
                for ne in neighbors:
                    ne_action = self.G.node[ne]["action"]
                    n_reward += self.payoff_matrix[n_action][ne_action][0]
                self.G.node[n]["reward"] = n_reward
                self.agent_payoff_table[n][i] = n_reward/len(neighbors)
                self.G.node[n]["agent"].update_q(0, n_reward)
