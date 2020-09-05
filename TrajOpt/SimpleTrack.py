import numpy as np 
import casadi as ca
from matplotlib import pyplot as plt

from TrajPlanner import *

track = np.array([[1, 1], 
                [1, 2], 
                [2, 3], 
                [2, 4], 
                [3, 5], 
                [4, 5], 
                [5, 5], 
                [6, 6], 
                [7, 6]])

ws = np.ones_like(track) * 1
track = np.concatenate([track, ws], axis=-1)

plot_track(track)

