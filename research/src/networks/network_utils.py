# -*- coding: utf-8 -*-
# author : Takuro Yamazaki
# description : ネットワークの定義


import networkx as nx
import matplotlib.pyplot as plt

from .grid_2d_graph import grid_2d_graph

class graph_generator(object):
    def __init__(self):
        pass

    @staticmethod
    def complete_graph(n=100):
        """完全グラフ
        :param n: node num
        :return: complete_graph instance
        """
        return nx.complete_graph(n)

    @staticmethod
    def random_graph(n=100, m=2500):
        """ランダムグラフ
        :param n: node num
        :param m: edge num
        :return: dense_gnm_random_graph instance
        """
        return nx.dense_gnm_random_graph(n, m)

    @staticmethod
    def grid_2d_graph(m=10, n=10):
        """2Dグリッドグラフ
        :param m: row num
        :param n: col num
        :return: grid_2d_graph instance
        """
        return grid_2d_graph(m, n)

    @staticmethod
    def grid_graph(dim=[5, 5, 4]):
        """nDグリッドグラフ
        :param dim: dimension
        :return grid_graph instance
        """
        return nx.grid_graph(dim)

    @staticmethod
    def watts_strogatz_graph(n=100, k=50, p=0.5):
        """ワッツストロガッツモデル
        :param n: node num
        :param k: neighbor num
        :param p: prob
        :return: watts_strogatz_graph instance
        """
        return nx.watts_strogatz_graph(n, k, p)

    # @staticmethod
    # def newman_watts_strogatz_graph(n=100, k=30, p=0.5):
    #     """ニューマンワッツストロガッツモデル(In contrast with watts_strogatz_graph(), no edges are removed)
    #     :param n: node num
    #     :param k: neighbor num
    #     :param p: prob
    #     :return: newman_watts_strogatz_graph instance
    #     """
    #     return nx.newman_watts_strogatz_graph(n, k, p)
    #
    # @staticmethod
    # def connected_watts_strogatz_graph(n=100, k=50, p=0.5):
    #     """連結ワッツストロガッツモデル(よくわからんから比較する)
    #     :param n: node num
    #     :param k: neighbor num
    #     :param p: prob
    #     :return: connected_watts_strogatz_graph instance
    #     """
    #     return nx.connected_watts_strogatz_graph(n, k, p)

    @staticmethod
    def barabasi_albert_graph(n=100, m=30):
        """バラバシアルバートモデル
        :param n: node num
        :param m: number of edges to attach from a new node to existing nodes
        :return: barabasi_albert_graph instance
        """
        return nx.barabasi_albert_graph(n, m)

    @staticmethod
    def powerlaw_cluster_graph(n=100, m=50, p=0.5):
        """powerlaw cluster graph
        :param n: node num
        :param m: the number of random edges to add for each new node
        :param p: prob
        :return: powerlaw_cluster_graph instance
        """
        return nx.powerlaw_cluster_graph(n, m, p)


class graph_utils(object):
    def __init__(self):
        pass

    @staticmethod
    def average_shortest_path_length(G):
        """平均経路長
        :param G: graph
        :return: float, average_shortest_path_length
        """
        return nx.average_shortest_path_length(G)

    @staticmethod
    def average_clustering(G):
        """クラスタリング係数
        :param G: graph
        :return: float, average_clustering
        """
        return nx.average_clustering(G)

    @staticmethod
    def assortativity(G):
        """次数相関(assortativity)
        ref : http://www.logos.ic.i.u-tokyo.ac.jp/~chik/InfoTech12/08%20Masuda.pdf
        :param G: graph
        :return: float, assortativity
        """
        # nx.degree_assorattivity_coefficientと同じだが，計算時間が早い
        return nx.degree_pearson_correlation_coefficient(G)

    @staticmethod
    def pagerank(G, alpha=0.85):
        """pagerank
        :param G: graph
        :param alpha: dampling parameter for pagerank
        :return: dict, key node, value pagerank
        """
        # nx.pagerankと同じだけどfastestかつmost accurateらしい
        return nx.pagerank_numpy(G)

    @staticmethod
    def degree_centrality(G):
        """次数中心性
        :param G: graph
        :return: float, degree centrality
        """
        return nx.degree_centrality(G)

    @staticmethod
    def closeness_centrality(G):
        """近接中心性
        :param G: graph
        :return: float, closeness centrality
        """
        return nx.closeness_centrality(G)

    @staticmethod
    def betweeness_centrality(G):
        """媒介中心性
        :param G: graph
        :return: float, betweeness centrality
        """
        return nx.betweenness_centrality(G)

    @staticmethod
    def plot_degree_histogram(G, f_name, title="Degree rank plot"):
        """plot loglog degree histgram
        :param G: graph
        :param f_name: str, file_path
        :return: None
        """
        degrees = sorted(nx.degree(G).values(), reverse=True)
        plt.loglog(degrees, "b-", alpha=0.6, marker='.')
        plt.title(title)
        plt.ylabel("degree")
        plt.xlabel("rank")

        plt.savefig(f_name, bbox_inches="tight")
        plt.clf()

    @staticmethod
    def plot_graph(G, f_name):
        """save graph figure
        :param G: graph
        :param f_name: str, file path
        :return: None
        """
        plt.figure(figsize=(15,15))
        pos = nx.spring_layout(G, k=0.8)

        node_size = [len(G.neighbors(n))*2 for n in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_color="royalblue", alpha=0.8, node_size=node_size)
        nx.draw_networkx_edges(G, pos, edge_color="gray", alpha=0.5)

        plt.axis("off")
        plt.savefig(f_name, bbox_inches="tight")
        plt.clf()

    @staticmethod
    def read_graph(f_name):
        """load graph from pickle
        :param f_name: str, file path
        :return: graph
        """
        G = nx.read_gpickle(f_name)
        return G

    @staticmethod
    def write_graph(G, f_name):
        """save graph in pickle
        :param G: graph
        :param f_name: str, file path, ###.gpickle
        :return: None
        """
        nx.write_gpickle(G, f_name)
