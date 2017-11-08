# -*- coding: utf-8 -*-
# author : Takuro Yamazaki
# description : 利得行列の定義

# TODO
# payoff matrixはnormarizeされる必要ある？
# Q learningの収束性の必要条件に含まれるっぽいことが"Lenient frequency adjusted Q-Learningに書いてある"


import numpy as np
import pandas as pd

__all__ = ["matching_pennies", "coodination_game", "stag_hunt","prisoners_dilemma", "social_dilemma_sen",
           "coodination_game_sen", "chicken_game", "tricky_game", "matching_pennies_sig", "coodination_game_sig",
           "stag_hunt_sig", "prisoners_dilemma_sig", "chicken_game_sig", "tricky_game_sig"]


def matching_pennies():
    return "matching_pennies", pd.DataFrame(np.array([[1, -1],[-1, 1]]), index=list('cd'), columns=list('cd'))


def coodination_game():
    return "coodination_game", pd.DataFrame(np.array([[2, 0], [0, 4]]), index=list('cd'), columns=list('cd'))


def stag_hunt():
    return "stag_hunt", pd.DataFrame(np.array([[2, 3], [-5, 4]]), index=list('cd'), columns=list('cd'))


def prisoners_dilemma():
    return "prisoners_dilemma", pd.DataFrame(np.array([[3, 0],[5, 1]]),index=list('cd'), columns=list('cd'))


def social_dilemma_sen():
    return "social dilemma sen", pd.DataFrame(np.array([[-1, 3],[2, 1]]),index=list('cd'), columns=list('cd'))

def coodination_game_sen():
    return "social dilemma sen", pd.DataFrame(np.array([[4, -1],[-1, 4]]),index=list('cd'), columns=list('cd'))


def chicken_game():
    return "chicken_game", pd.DataFrame(np.array([[3, 2], [3.5, 1]]), index=list('cd'), columns=list('cd'))


def tricky_game():
    return "tricky_game", pd.DataFrame(np.array([[0, 3], [1, 2]]), index=list('cd'), columns=list('cd'))


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
