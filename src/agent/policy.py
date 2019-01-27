#-*- coding: utf-8 -*-
# author : Takuro Yamazaki
# description : エージェントの行動選択手法の定義

# TODO
#   - 温度パラメータの管理

# Reference
#   - マルチエージェント学習, 高玉圭樹

import numpy as np

__all__ = ["softmax_boltzman", "softmax_roulette", "eps_greedy", "greedy"]

def softmax_boltzman(q_row, T=1.0):
    """
    :description ボルツマンソフトマックス方策, for off policy learning
    :param q_row: q_table row of object state s
    :param T: temperature parameter
    :return selected action arg
    """
    obj = np.arange(len(q_row))
    q_row = np.array(q_row)
    ep = np.exp(q_row/T)
    act = np.random.choice(obj, p=ep/np.sum(ep))
    pr = (ep/np.sum(ep))[act]
    return act, pr


def eps_greedy(q_row, eps=0.1):
    """
    :description epsilon greedy 方策
    :param q_row: q_table row of object state s
    :param eps: epsilon
    :return selected action arg
    """
    if np.random.rand() < eps:
        obj = q_row.index
        return np.random.choice(obj) #randomに戦略を返す
    else:
        q_index = q_row.index
        index = np.random.choice(np.argwhere(q_row == np.amax(q_row)).flatten())
        return q_index[index] #greedyな戦略


def greedy(q_row):
    """
    :description greedy 方策
    :param q_row: q_table row of object state s
    :return selected action arg
    """
    return q_row.argmax()


def softmax_roulette(q_row):
    """
    :description ルーレットソフトマックス方策
    :param q_row: q_table row of object state s
    :return selected action arg
    """
    obj = np.arange(q_row.size)
    q_row = np.array(q_row)
    try: #Q値がマイナスをとるときは適用不可能
        return np.random.choice(obj, p=q_row/np.sum(q_row))
    except ValueError as e:
        print("Exception occuered!!")
        print("{0}".format(e))
