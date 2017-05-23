#-*- coding: utf-8 -*-
# author : Takuro Yamazaki
# last update : 05/23/2017
# description : 環境中のエージェントのつながり方を定義するネットワークモデル

# TODO
#   - adaptive or temporalな変化の実装
#   - 一方向のリンク(有向)
#   - 座標指定のdraw

# Reference
#   - 他データ形式とnetworkx.Graphの変換(http://networkx.readthedocs.io/en/stable/_modules/networkx/convert_matrix.html)

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
plt.style.use('ggplot')

class Network(object):
    def __init__(self, G):
        self.G = graph

    def to_matrix(self, *args, **kwargs):
        """
        @description Graph -> Numpy matrixの隣接行列に変換
        @param networkxのGraph.to_numpy_matrixに準拠
        @return numpy matrixの隣接行列
        """
        return self.G.to_numpy_matrix(*args, **kwargs)

    def get_adjacent(self, node_id):
        """
        @description あるノードの隣接要素を返す
        @param node_id 対象ノード
        @return 対象ノードの隣接要素list
        """
        return self.G.neighbors(node_id)

    def from_matrix(self, np_mat):
        """
        @description Numpy matrix -> Graphの変換
        @param np_mat Numpy matrixの隣接行列
        """
        self.G = nx.from_numpy_matrix(np_mat)

    def draw(self, f_name, w_degree=False):
        """
        @description Graphの可視化
        @param f_name save file name
        @param w_degree draw graph dependent on the node degree
        """
        #行動によって色分けしてるけど別ので分類するなら変えよう
        color = ["r" if self.G.node[n]["action"] == 0 else "b" if self.G.node[n]["action"] == 1 else "g" for n in self.G.nodes()]
        if w_degree:
            d = nx.degree(G)
            nx.draw(self.G, pos=nx.circular_layout(self.G), nodelist=d.keys(), node_color=color, node_size=[50+v*30 for v in d.values()])
        else:
            nx.draw(self.G, pos=nx.circular_layout(self.G), node_color=color)
        plt.savefig(f_name)
