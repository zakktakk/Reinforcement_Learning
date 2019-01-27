# -*- coding: utf-8 -*-
# author : Takuro Yamazaki
# description : 利得行列の定義

import numpy as np
import pandas as pd

__all__ = ["matching_pennies", "coodination_game", "stag_hunt","prisoners_dilemma", "chicken_game", "tricky_game",
           "CG", "PG", "PSCG", "FSCG","matching_pennies_sig", "coodination_game_sig",
           "stag_hunt_sig", "prisoners_dilemma_sig", "chicken_game_sig", "tricky_game_sig"]


def matching_pennies():
    return "matching_pennies", pd.DataFrame(np.array([[1, -1],[-1, 1]]), index=list('cd'), columns=list('cd'))


def coodination_game():
    return "coodination_game", pd.DataFrame(np.array([[2, 0], [0, 4]]), index=list('cd'), columns=list('cd'))


def stag_hunt():
    return "stag_hunt", pd.DataFrame(np.array([[2, 3], [-5, 4]]), index=list('cd'), columns=list('cd'))


def prisoners_dilemma():
    return "prisoners_dilemma", pd.DataFrame(np.array([[3, 0],[5, 1]]),index=list('cd'), columns=list('cd'))


def chicken_game():
    return "chicken_game", pd.DataFrame(np.array([[3, 2], [3.5, 1]]), index=list('cd'), columns=list('cd'))


def tricky_game():
    return "tricky_game", pd.DataFrame(np.array([[0, 3], [1, 2]]), index=list('cd'), columns=list('cd'))


def CG():
    return "CG", pd.DataFrame(np.array([[11, -30, 0], [-30, 7, 6], [0, 0, 5]]), index=list('abc'), columns=list('abc'))


def PG(k=-50):
    return "PG", pd.DataFrame(np.array([[10, 0, k], [0, 2, 0], [k, 0, 10]]), index=list('abc'), columns=list('abc'))


def PSCG():
    return "PSCG", pd.DataFrame(np.array([[11, -30, 0], [-30, (14, 0), 6], [0, 0, 5]]),
                                index=list('abc'), columns=list('abc'))


def FSCG():
    return "FSCG", pd.DataFrame(np.array([[(10, 12), (5, -65), (8, -8)], [(5, -65), (14, 0), (12, 0)], [(5, -5), (5, -5), (10, 0)]]),
                            index=list('abc'), columns=list('abc'))


###########payoff matrix for signaling###########
# 無コストのシグナル
def prisoners_dilemma_sig():
    return "prisoners_dilemma_sig", pd.DataFrame(np.array([[3, 3, 0, 0],
                                                           [3, 3, 0, 0],
                                                           [5, 5, 1, 1],
                                                           [5, 5, 1, 1]]),
                                                index=['cs', 'c', 'ds', 'd'], columns=['cs', 'c', 'ds', 'd'])


def matching_pennies_sig():
    return "matching_pennies_sig", pd.DataFrame(np.array([[1, 1, -1, -1],
                                                          [1, 1, -1, -1],
                                                          [-1, -1, 1, 1],
                                                          [-1, -1, 1, 1]]),
                                                index=['cs', 'c', 'ds', 'd'], columns=['cs', 'c', 'ds', 'd'])

def coodination_game_sig():
    return "coodination_game_sig", pd.DataFrame(np.array([[2, 2, 0, 0],
                                                          [2, 2, 0, 0],
                                                          [0, 0, 4, 4],
                                                          [0, 0, 4, 4]]),
                                                index = ['cs', 'c', 'ds', 'd'], columns = ['cs', 'c', 'ds', 'd'])


def stag_hunt_sig():
    return "stag_hunt_sig", pd.DataFrame(np.array([[2, 2, 3, 3],
                                                   [2, 2, 3, 3],
                                                   [-5, -5, 4, 4],
                                                   [-5, -5, 4, 4]]),
                                         index=['cs', 'c', 'ds', 'd'], columns=['cs', 'c', 'ds', 'd'])


def chicken_game_sig():
    return "chicken_game_sig", pd.DataFrame(np.array([[3, 3, 2, 2],
                                                      [3, 3, 2, 2],
                                                      [3.5, 3.5, 1, 1],
                                                      [3.5, 3.5, 1, 1]]),
                                            index=['cs', 'c', 'ds', 'd'], columns=['cs', 'c', 'ds', 'd'])


def tricky_game_sig():
    return "tricky_game_sig", pd.DataFrame(np.array([[0, 0, 3, 3],
                                                     [0, 0, 3, 3],
                                                     [1, 1, 2, 2],
                                                     [1, 1, 2, 2]]),
                                           index=['cs', 'c', 'ds', 'd'], columns=['cs', 'c', 'ds', 'd'])
