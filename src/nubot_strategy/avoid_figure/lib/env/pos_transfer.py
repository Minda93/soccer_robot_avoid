#!/usr/bin/env python3
# -*- coding: utf-8 -*-+

""" tool """
import numpy as np

r""" soccer field
    width : 700
    height : 500
"""

SOCCOR_CENTER_X = 350
SOCCOR_CENTER_Y = 250

def Pos_Draw2Soccer(pos):
    x = pos[0] - SOCCOR_CENTER_X
    y = pos[1] - SOCCOR_CENTER_Y
    return np.array([x,y])

def Pos_Soccer2Draw(pos):
    x = pos[0] + SOCCOR_CENTER_X
    y = pos[1] + SOCCOR_CENTER_Y
    return np.array([x,y])