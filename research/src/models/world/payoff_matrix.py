# -*- coding: utf-8 -*-
# author : Takuro Yamazaki
# description : 利得行列の定義

# TODO
# payoff matrixはnormarizeされる必要ある？
# Q learningの収束性の必要条件に含まれるっぽいことが"Lenient frequency adjusted Q-Learningに書いてある"

# Reference
#   - J.W.Crandall, M.A.Goodrich, "Learning to compete, compromise, and cooperate in repeated general-sum games", In ICML, 2005

import numpy as np

__all__ = ["rock_paper_scissors", "shapleys_game", "matching_pennies", "coodination_game", "stag_hunt",
           "prisoners_dilemma", "chicken_game", "tricky_game", "matching_pennies_sig", "coodination_game_sig",
           "stag_hunt_sig", "prisoners_dilemma_sig", "chicken_game_sig", "tricky_game_sig"]


def rock_paper_scissors():
    return "rock_paper_scissors", {'r': [(0, 0), (-1, 1), (1, -1)],
                                   'p': [(1, -1), (0, 0), (-1, 1)],
                                   's': [(-1, 1), (1, -1), (0, 0)]}


def shapleys_game():
    return "shapleys_game", {'a': [(0, 0), (0, 1), (1, 0)],
                             'b': [(1, 0), (0, 0), (0, 1)],
                             'c': [(0, 1), (1, 0), (0, 0)]}


def matching_pennies():
    return "matching_pennies", {'a': [(1, -1), (-1, 1)],
                                'b': [(-1, 1), (1, -1)]}


def coodination_game():
    return "coodination_game", {'a': [(2, 2), (0, 0)],
                                'b': [(0, 0), (4, 4)]}


def stag_hunt():
    return "stag_hunt", {'a': [(2, 2), (3, -5)],
                         'b': [(-5, 3), (4, 4)]}


def prisoners_dilemma():
    return "prisoners_dilemma", {'c': [(3, 3), (0, 5)],
                                 'd': [(5, 0), (1, 1)]}


def chicken_game():
    return "chicken_game", {'c': [(3, 3), (2, 3.5)],
                            'd': [(3.5, 2), (1, 1)]}


def tricky_game():
    return "tricky_game", {'a': [(0, 3), (3, 2)],
                           'b': [(1, 0), (2, 1)]}


###########payoff matrix for signaling###########
# 無コストのシグナル
def prisoners_dilemma_sig():
    return "prisoners_dilemma_sig", {'cs': [(3, 3), (3, 3), (0, 5), (0, 5)],
                                     'c': [(3, 3), (3, 3), (0, 5), (0, 5)],
                                     'ds': [(5, 0), (5, 0), (1, 1), (1, 1)],
                                     'd': [(5, 0), (5, 0), (1, 1), (1, 1)]}


def matching_pennies_sig():
    return "matching_pennies_sig", {'as': [(1, -1), (1, -1), (-1, 1), (-1, 1)],
                                    'a': [(1, -1), (1, -1), (-1, 1), (-1, 1)],
                                    'bs': [(-1, 1), (-1, 1), (1, -1), (1, -1)],
                                    'b': [(-1, 1), (-1, 1), (1, -1), (1, -1)]}


def coodination_game_sig():
    return "coodination_game_sig", {'as': [(2, 2), (2, 2), (0, 0), (0, 0)],
                                    'a': [(2, 2), (2, 2), (0, 0), (0, 0)],
                                    'bs': [(0, 0), (0, 0), (4, 4), (4, 4)],
                                    'b': [(0, 0), (0, 0), (4, 4), (4, 4)]}


def stag_hunt_sig():
    return "stag_hunt_sig", {'as': [(2, 2), (2, 2), (3, -5), (3, -5)],
                             'a': [(2, 2), (2, 2), (3, -5), (3, -5)],
                             'bs': [(-5, 3), (-5, 3), (4, 4), (4, 4)],
                             'b': [(-5, 3), (-5, 3), (4, 4), (4, 4)]}


def chicken_game_sig():
    return "chicken_game_sig", {'cs': [(3, 3), (3, 3), (2, 3.5), (2, 3.5)],
                                'c': [(3, 3), (3, 3), (2, 3.5), (2, 3.5)],
                                'ds': [(3.5, 2), (3.5, 2), (1, 1), (1, 1)],
                                'd': [(3.5, 2), (3.5, 2), (1, 1), (1, 1)]}


def tricky_game_sig():
    return "tricky_game_sig", {'as': [(0, 3), (0, 3), (3, 2), (3, 2)],
                               'a': [(0, 3), (0, 3), (3, 2), (3, 2)],
                               'bs': [(1, 0), (1, 0), (2, 1), (2, 1)],
                               'b': [(1, 0), (1, 0), (2, 1), (2, 1)]}
