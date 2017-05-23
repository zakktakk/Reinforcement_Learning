#-*- coding: utf-8 -*-
# author : Takuro Yamazaki
# last update : 05/23/2017
# description : エージェントの行動選択手法の定義

# TODO
#   - 温度パラメータの管理

# Reference
#   - マルチエージェント学習, 高玉圭樹

import numpy as np
import warning

__all__ = ["softmax_boltzman", "softmax_roulette", "eps_greedy", "greedy"]

def softmax_boltzman(q_row, T=1.0):
    """
    @description ボルツマンソフトマックス方策
    @param q_row q_table row of object state s
    @param T temperature parameter
    @return selected action arg
    """
    notnan_idx = np.invert(np.isnan(q_row))
    obj = np.arange(len(q_row))[notnan_idx] #Noneじゃない要素
    q_row = q_row[notnan_idx]
    return np.random.choice(obj, p=(np.exp(q_row/T))/np.sum(np.exp(q_row/T)))

def eps_greedy(q_row, eps=0.1):
    """
    @description epsilon greedy 方策
    @param q_row q_table row of object state s
    @param eps epsilon
    @return selected action arg
    """
    q_row = np.array(q_row)
    if np.random.rand() < eps:
        obj = np.arange(len(q_row))[np.invert(np.isnan(q_row))] #Noneじゃない要素
        return np.random.choice(obj) #randomに戦略を返す
    else:
        return np.nanargmax(q_row) #greedyな戦略

def greedy(q_row):
    """
    @description greedy 方策
    @param q_row q_table row of object state s
    @return selected action arg
    """
    q_row = np.array(q_row)
    return np.nanargmax(q_row)

def softmax_roulette(q_row):
    """
    @description ルーレットソフトマックス方策
    @param q_row q_table row of object state s
    @return selected action arg
    """
    notnan_idx = np.invert(np.isnan(q_row))
    obj = np.arange(len(q_row))[notnan_idx]
    q_row = q_row[notnan_idx]
    try: #Q値がマイナスをとるときは適用不可能
        return np.random.choice(obj, p=q_row/np.sum(q_row))
    except ValueError as e:
        print("Exception occuered!!")
        print("{0}".format(e))
