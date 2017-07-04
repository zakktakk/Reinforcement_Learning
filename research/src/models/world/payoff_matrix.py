#-*- coding: utf-8 -*-
# author : Takuro Yamazaki
# last update : 05/23/2017
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
    return "rock_paper_scissors", np.array(['r', 'p', 's']), np.array([[(0,0),(-1,1),(1,-1)],[(1,-1),(0,0),(-1,1)],[(-1,1),(1,-1),(0,0)]])

def shapleys_game():
    return "shapleys_game", np.array(['a', 'b', 'c']), np.array([[(0,0),(0,1),(1,0)],[(1,0),(0,0),(0,1)],[(0,1),(1,0),(0,0)]])

def matching_pennies():
    return "matching_pennies", np.array(['a', 'b']), np.array([[(1,-1), (-1,1)],[(-1,1), (1,-1)]])

def coodination_game():
    return "coodination_game", np.array(['a', 'b']),np.array([[(2,2), (0,0)],[(0,0), (4,4)]])

def stag_hunt():
    return "stag_hunt", np.array(['a', 'b']),np.array([[(2,2), (3,-5)],[(-5,3), (4,4)]])

def prisoners_dilemma():
    return "prisoners_dilemma", np.array(['c', 'd']),np.array([[(3,3), (0,5)],[(5,0), (1,1)]])

def chicken_game():
    return "chicken_game", np.array(['c', 'd']),np.array([[(3,3), (2,3.5)],[(3.5,2), (1,1)]])

def tricky_game():
    return "tricky_game", np.array(['a', 'b']),np.array([[(0,3), (3,2)],[(1,0), (2,1)]])


###########payoff matrix for signaling###########
# 無コストのシグナル
def prisoners_dilemma_sig():
    return "prisoners_dilemma_sig", np.array(['cs', 'c', 'ds', 'd']),np.array([[(3,3),(3,3),(0,5),(0,5)],[(3,3),(3,3),(0,5),(0,5)],[(5,0),(5,0),(1,1),(1,1)],[(5,0),(5,0),(1,1),(1,1)]])

def matching_pennies_sig():
    return "matching_pennies_sig", np.array(['as', 'a', 'bs', 'b']), np.array([[(1,-1),(1,-1),(-1,1),(-1,1)],[(1,-1),(1,-1),(-1,1),(-1,1)],[(-1,1),(-1,1),(1,-1),(1,-1)],[(-1,1),(-1,1),(1,-1),(1,-1)]])

def coodination_game_sig():
    return "coodination_game_sig", np.array(['as', 'a', 'bs', 'b']),np.array([[(2,2),(2,2),(0,0),(0,0)],[(2,2),(2,2),(0,0),(0,0)],[(0,0),(0,0),(4,4),(4,4)],[(0,0),(0,0),(4,4),(4,4)]])

def stag_hunt_sig():
    return "stag_hunt_sig", np.array(['as', 'a', 'bs', 'b']),np.array([[(2,2),(2,2),(3,-5),(3,-5)],[(2,2),(2,2),(3,-5),(3,-5)],[(-5,3),(-5,3),(4,4),(4,4)],[(-5,3),(-5,3),(4,4),(4,4)]])

def chicken_game_sig():
    return "chicken_game_sig", np.array(['cs', 'c', 'ds', 'd']),np.array([[(3,3),(3,3),(2,3.5),(2,3.5)],[(3,3),(3,3),(2,3.5),(2,3.5)],[(3.5,2),(3.5,2),(1,1),(1,1)],[(3.5,2),(3.5,2),(1,1),(1,1)]])

def tricky_game_sig():
    return "tricky_game_sig", np.array(['as', 'a', 'bs', 'b']),np.array([[(0,3),(0,3),(3,2),(3,2)],[(0,3),(0,3),(3,2),(3,2)],[(1,0),(1,0),(2,1),(2,1)],[(1,0),(1,0),(2,1),(2,1)]])
