import numpy as np 
import matplotlib.pyplot as plt 
import csv
import math
import casadi as ca

import os
import sys
import time

import LibFunctions as lib 
# from RaceTrackMaps import RaceMap
# from PathFinder import PathFinder


def create_track():
    track_name = 'RaceTrack1000'
    env_map = RaceMap('RaceTrack1000')
    env_map.reset_map()
    fcn = env_map.obs_free_hm._check_line
    path_finder = PathFinder(fcn, env_map.start, env_map.end)

    path = path_finder.run_search(5)
    # env_map.race_course.show_map(path=path, show=True)

    width = 5
    widths = np.ones_like(path) * width
    track = np.concatenate([path, widths], axis=-1)

    filename = 'Maps/' + track_name + '_abscissa.csv'

    with open(filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        # csvwriter.writerow(['xi', 'yi', 'nl', 'nr'])
        csvwriter.writerows(track)

    print(f"Track Created")


def load_track(filename='Maps/RaceTrack1000_abscissa.csv', show=True):
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

def interp_track(track, step_size=1):
    ref_track_closed = np.vstack((track, track[0]))

    seg_lengths = np.sqrt(np.sum(np.power(np.diff(ref_track_closed[:, :2], axis=0), 2), axis=1))
    dists_cum = np.cumsum(seg_lengths)
    dists_cum = np.insert(dists_cum, 0, 0.0)

    no_points_interp = math.ceil(dists_cum[-1] / step_size) + 1
    dists_interp = np.linspace(0.0, dists_cum[-1], no_points_interp)

    interp_track = np.zeros((no_points_interp, 4))
    interp_track[:, 0] = np.interp(dists_interp, dists_cum, ref_track_closed[:, 0])
    interp_track[:, 1] = np.interp(dists_interp, dists_cum, ref_track_closed[:, 1])
    interp_track[:, 2] = np.interp(dists_interp, dists_cum, ref_track_closed[:, 2])
    interp_track[:, 3] = np.interp(dists_interp, dists_cum, ref_track_closed[:, 3])

    return interp_track

def generate_bounds(track:np.ndarray, normvec):
    lx = track[:, 0] - track[:, 2] * normvec[:, 0]
    ly = track[:, 1] - track[:, 2] * normvec[:, 1]
    rx = track[:, 0] + track[:, 3] * normvec[:, 0]
    ry = track[:, 1] + track[:, 3] * normvec[:, 1]

    l = np.concatenate((lx[:, None], ly[:, None]), axis=-1)
    r = np.concatenate((rx[:, None], ry[:, None]), axis=-1)
    
    return l, r

def generate_raceline(track:np.ndarray, n_set, normvec):
    lx = track[:, 0] + n_set[:, 0] * normvec[:, 0]
    ly = track[:, 1] + n_set[:, 0] * normvec[:, 1]

    l = np.concatenate((lx[:, None], ly[:, None]), axis=-1)
    
    return l

def calc_splines(path):
    path = np.vstack((path, path[0, :]))
    seg_lengths = np.sqrt(np.sum(np.power(np.diff(path, axis=0), 2), axis=1))

    no_splines = path.shape[0] - 1

    M = np.zeros((no_splines * 4, no_splines * 4))
    b_x = np.zeros((no_splines * 4, 1))
    b_y = np.zeros((no_splines * 4, 1))

    scaling = np.ones(no_splines - 1)

    template_M = np.array(                          # current point               | next point              | bounds
                [[1,  0,  0,  0,  0,  0,  0,  0],   # a_0i                                                  = {x,y}_i
                 [1,  1,  1,  1,  0,  0,  0,  0],   # a_0i + a_1i +  a_2i +  a_3i                           = {x,y}_i+1
                 [0,  1,  2,  3,  0, -1,  0,  0],   # _      a_1i + 2a_2i + 3a_3i      - a_1i+1             = 0
                 [0,  0,  2,  6,  0,  0, -2,  0]])  # _             2a_2i + 6a_3i               - 2a_2i+1   = 0

    for i in range(no_splines):
        j = i * 4

        if i < no_splines - 1:
            M[j: j + 4, j: j + 8] = template_M

            M[j + 2, j + 5] *= scaling[i]
            M[j + 3, j + 6] *= math.pow(scaling[i], 2)

        else:
            # no curvature and heading bounds on last element (handled afterwards)
            M[j: j + 2, j: j + 4] = [[1,  0,  0,  0],
                                     [1,  1,  1,  1]]

        b_x[j: j + 2] = [[path[i,     0]],
                         [path[i + 1, 0]]]
        b_y[j: j + 2] = [[path[i,     1]],
                         [path[i + 1, 1]]]

    M[-2, 1] = scaling[-1]
    M[-2, -3:] = [-1, -2, -3]
    # b_x[-2] = 0
    # b_y[-2] = 0

    # curvature boundary condition (for a closed spline)
    M[-1, 2] = 2 * math.pow(scaling[-1], 2)
    M[-1, -2:] = [-2, -6]
    # b_x[-1] = 0
    # b_y[-1] = 0

    x_les = np.squeeze(np.linalg.solve(M, b_x))  # squeeze removes single-dimensional entries
    y_les = np.squeeze(np.linalg.solve(M, b_y))

    # get coefficients of every piece into one row -> reshape
    coeffs_x = np.reshape(x_les, (no_splines, 4))
    coeffs_y = np.reshape(y_les, (no_splines, 4))

    # get normal vector (behind used here instead of ahead for consistency with other functions) (second coefficient of
    # cubic splines is relevant for the heading)
    normvec = np.stack((coeffs_y[:, 1], -coeffs_x[:, 1]), axis=1)

    # normalize normal vectors
    norm_factors = 1.0 / np.sqrt(np.sum(np.power(normvec, 2), axis=1))
    normvec_normalized = np.expand_dims(norm_factors, axis=1) * normvec

    return coeffs_x, coeffs_y, normvec_normalized

def calc_my_nvecs(track):
    normvec = []
    th = lib.get_bearing(track[0, 0:2], track[1, 0:2])
    normvec.append(lib.theta_to_xy(th + np.pi/2))
    for i in range(len(track) - 2):
        th0 = lib.get_bearing(track[i, 0:2], track[i+1, 0:2])
        th1 = lib.get_bearing(track[i+1, 0:2], track[i+2, 0:2])
        th = (th0 + th1) / 2
        normvec.append(lib.theta_to_xy(th + np.pi/2))
    
    th = lib.get_bearing(track[-2, 0:2], track[-1, 0:2])
    normvec.append(lib.theta_to_xy(th + np.pi/2)) # last point
    normvec = np.array(normvec)

    return normvec

def plot_spline_track(track, wait=False):
    nvs = calc_my_nvecs(track)
    c_line = track[:, 0:2]
    l_line = c_line - np.array([nvs[:, 0] * track[:, 2], nvs[:, 1] * track[:, 2]]).T
    r_line = c_line + np.array([nvs[:, 0] * track[:, 3], nvs[:, 1] * track[:, 3]]).T

    plt.figure(1)
    plt.plot(c_line[:, 0], c_line[:, 1], linewidth=2)
    plt.plot(l_line[:, 0], l_line[:, 1], linewidth=1)
    plt.plot(r_line[:, 0], r_line[:, 1], linewidth=1)

    if wait:
        plt.show()

def plot_track(track, wait=False):
    # track = np.vstack((track, track[-1, :]))

    # x, y, normvec = calc_splines(track)
    normvec = calc_my_nvecs(track)
    l, r = generate_bounds(track, normvec)


    plt.figure()

    for i in range(len(l)):
        xs = [l[i, 0], r[i, 0]]
        ys = [l[i, 1], r[i, 1]]
        plt.plot(xs, ys, '--')
    plt.plot(track[:, 0], track[:, 1], linewidth=2)
    plt.plot(l[:, 0], l[:, 1], linewidth=0.8)
    plt.plot(r[:, 0], r[:, 1], linewidth=0.8)

    plt.pause(0.001)
    if wait:
        plt.show()

def plot_race_line(track, n_set, wait=False):
    # track = np.vstack((track, track[-1, :]))

    # x, y, normvec = calc_splines(track)
    normvec = calc_my_nvecs(track)
    l, r = generate_bounds(track, normvec)

    plt.figure()

    for i in range(len(l)):
        xs = [l[i, 0], r[i, 0]]
        ys = [l[i, 1], r[i, 1]]
        plt.plot(xs, ys, '--')
    
    plt.pause(0.001)

    plt.plot(track[:, 0], track[:, 1], linewidth=2)
    plt.plot(l[:, 0], l[:, 1], linewidth=0.5)
    plt.plot(r[:, 0], r[:, 1], linewidth=0.5)

    plt.pause(0.001)

    line = generate_raceline(track, n_set, normvec)
    plt.plot(line[:, 0], line[:, 1], linewidth=4)

    plt.pause(0.001)
    if wait:
        plt.show()


def add_points(track, points_add):
    N = len(track)
    new_track = np.zeros(((N-1) * points_add + 1, 4))
    for i in range(N-1):
        # new_track[i * points_add] = track[i]
        for j in range(points_add):
            new_x = (-track[i, 0] + track[i+1, 0]) * j/points_add + track[i, 0]
            new_y = (-track[i, 1] + track[i+1, 1]) * j/points_add + track[i, 1]
            new_l = (-track[i, 2] + track[i+1, 2]) * j/points_add + track[i, 2]
            new_r = (-track[i, 3] + track[i+1, 3]) * j/points_add + track[i, 3]
            new_pt = np.array([new_x, new_y, new_l, new_r])
            new_track[i * points_add + j] = new_pt

    new_track[-1] = track[-1]



    return new_track

def change_widths(track):
    w = track[0, 2] + track[0, 3]
    N = len(track)

    new_track = np.zeros_like(track, dtype=float)
    new_track[:, 0:2] = track[:, 0:2]
    new_track[0, 2:4] = track[0, 2:4]
    new_track[-1, 2:4] = track[-1, 2:4]

    for i in range(N-2):
        b0 = lib.get_bearing(track[i, 0:2], track[i+1, 0:2])
        b1 = lib.get_bearing(track[i+1, 0:2], track[i+2, 0:2])
        ddth = b0 - b1
        l =  w/2 + w/2 * ddth /(np.pi/2) 

        r = w - l
        new_track[i+1, 2] = r
        new_track[i+1, 3] = l

    plot_track(new_track)

    return new_track



def fix_up_track(track, points_add=5):
    # track = x, y, wl, wr

    track = change_widths(track)

    track = add_points(track, points_add)

    return track


def calc_spline_lengths(coeffs_x, coeffs_y):
    no_splines = coeffs_x.shape[0]
    spline_lengths = np.zeros(no_splines)
    no_interp_points = 15

    t_steps = np.linspace(0.0, 1.0, no_interp_points)
    spl_coords = np.zeros((no_interp_points, 2))

    for i in range(no_splines):
        spl_coords[:, 0] = coeffs_x[i, 0] \
                            + coeffs_x[i, 1] * t_steps \
                            + coeffs_x[i, 2] * np.power(t_steps, 2) \
                            + coeffs_x[i, 3] * np.power(t_steps, 3)
        spl_coords[:, 1] = coeffs_y[i, 0] \
                            + coeffs_y[i, 1] * t_steps \
                            + coeffs_y[i, 2] * np.power(t_steps, 2) \
                            + coeffs_y[i, 3] * np.power(t_steps, 3)

        spline_lengths[i] = np.sum(np.sqrt(np.sum(np.power(np.diff(spl_coords, axis=0), 2), axis=1)))

    return spline_lengths







if __name__ == "__main__":
    # create_track()
    track = load_track(show=False)

    track = interp_track(track)

    x, y, nvec = calc_splines(track)

    plot_track(track)

    opt_mintime(track, x, y, nvec)




