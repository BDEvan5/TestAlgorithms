import numpy as np 
import casadi as ca
from matplotlib import pyplot as plt
import math
import csv

import LibFunctions as lib 


def load_track(filename='TrajOpt/RaceTrack1000_abscissa.csv', show=True):
    track = []
    with open(filename, 'r') as csvfile:
        csvFile = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  
    
        for lines in csvFile:  
            track.append(lines)

    track = np.array(track)

    print(f"Track Loaded")

    return track


def interp_track(track):
    N = 100 
    seg_lengths = np.sqrt(np.sum(np.power(np.diff(track[:, :2], axis=0), 2), axis=1))
    dists_cum = np.cumsum(seg_lengths)
    dists_cum = np.insert(dists_cum, 0, 0.0)
    length = sum(seg_lengths)

    ds = length / N

    no_points_interp = math.ceil(dists_cum[-1] / (ds/5)) + 1
    dists_interp = np.linspace(0.0, dists_cum[-1], no_points_interp)

    interp_track = np.zeros((no_points_interp, 2))
    interp_track[:, 0] = np.interp(dists_interp, dists_cum, track[:, 0])
    interp_track[:, 1] = np.interp(dists_interp, dists_cum, track[:, 1])

    return interp_track

def get_nvec(x0, x2):
    th = lib.get_bearing(x0, x2)
    new_th = th + np.pi/2
    nvec = lib.theta_to_xy(new_th)

    return nvec

def check_nvec_cross(p1, v1, pt2, v2, w):
    e1 = lib.add_locations(p1, v1*w)
    e2 = lib.add_locations(p2, v2*w)


    e1 = lib.sub_locations(p1, v1*w)
    e2 = lib.sub_locations(p2, v2*w)


def create_nvecs(track):
    N = 100
    seg_lengths = np.sqrt(np.sum(np.power(np.diff(track[:, :2], axis=0), 2), axis=1))
    length = sum(seg_lengths)
    ds = length / N

    new_track, nvecs = [], []
    new_track.append(track[0, :])
    nvecs.append(get_nvec(track[0, :], track[1, :]))
    s = 0
    for i in range(len(track)-1):
        s = lib.get_distance(new_track[-1], track[i, :])
        if s > ds:
            nvec = get_nvec(new_track[-1], track[min((i+5, len(track)-1)), :])
            nvecs.append(nvec)
            new_track.append(track[i])

    return_track = np.concatenate([new_track, nvecs], axis=-1)

    return return_track

def plot_track(track, wait=False):
    c_line = track[:, 0:2]
    width = 5
    l_line = c_line - np.array([track[:, 2] * width, track[:, 3] * width]).T
    r_line = c_line + np.array([track[:, 2] * width, track[:, 3] * width]).T

    plt.figure(1)
    plt.plot(c_line[:, 0], c_line[:, 1], linewidth=2)
    plt.plot(l_line[:, 0], l_line[:, 1], linewidth=1)
    plt.plot(r_line[:, 0], r_line[:, 1], linewidth=1)

    plt.pause(0.0001)
    if wait:
        plt.show()


def run_map_gen():
    path = load_track()
    path = interp_track(path)
    track = create_nvecs(path)
    plot_track(track, True)


if __name__ == "__main__":
    run_map_gen()

