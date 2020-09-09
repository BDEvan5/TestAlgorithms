import numpy as np 
import casadi as ca
from matplotlib import pyplot as plt


def load_track(filename='Maps/RaceTrack2000_abscissa.csv', show=True):
    track = []
    with open(filename, 'r') as csvfile:
        csvFile = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  
    
        for lines in csvFile:  
            track.append(lines)

    track = np.array(track)

    print(f"Track Loaded")

    if show:
        plot_track(track)

    return track




