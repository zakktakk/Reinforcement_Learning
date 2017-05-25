#-*- coding: utf-8 -*-
# author : Takuro Yamazaki
# last update : 05/23/2017
# description : 利得行列の定義

# Reference
#   - J.W.Crandall, M.A.Goodrich, "Learning to compete, compromise, and cooperate in repeated general-sum games", In ICML, 2005

import numpy as np
__all__ = ["rock_paper_scissors", "shapleys_game", "matching_pennies", "coodination_game", "stag_hunt", "prisoners_dilemma", "chicken_game", "tricky_game"]

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
