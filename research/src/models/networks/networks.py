#-*- coding: utf-8 -*-
# author : Takuro Yamazaki
# last update : 05/23/2017
# description : 環境中のエージェントのつながり方を定義するネットワークモデル

# TODO
#   - adaptive or temporalな変化の実装
#   - 一方向のリンク(有向)

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
        """
        return self.G.to_numpy_matrix(*args, **kwargs)

    def from_matrix(self, np_mat):
        """
        @description Numpy matrix -> Graphの変換
        @param np_mat Numpy matrixの隣接行列
        """
        self.G = nx.from_numpy_matrix(np_mat)

    def visualize(self):
        nx.draw()
