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


def interp_track(track, N):
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

def plot_race_line(track, nset=None, wait=False):
    c_line = track[:, 0:2]
    l_line = c_line - np.array([track[:, 2] * track[:, 4], track[:, 3] * track[:, 4]]).T
    r_line = c_line + np.array([track[:, 2] * track[:, 5], track[:, 3] * track[:, 5]]).T

    plt.figure(1)
    plt.plot(c_line[:, 0], c_line[:, 1], linewidth=2)
    plt.plot(l_line[:, 0], l_line[:, 1], linewidth=1)
    plt.plot(r_line[:, 0], r_line[:, 1], linewidth=1)

    if nset is not None:
        deviation = np.array([track[:, 2] * nset[:, 0], track[:, 3] * nset[:, 0]]).T
        r_line = track[:, 0:2] + deviation
        plt.plot(r_line[:, 0], r_line[:, 1], linewidth=3)

    plt.pause(0.0001)
    if wait:
        plt.show()

def set_widths(track, width=5):
    N = len(track)

    ls, rs = [width], [width]
    for i in range(N-2):
        ls.append(width)
        rs.append(width)

    ls.append(width)
    rs.append(width)

    ls = np.array(ls)
    rs = np.array(rs)

    new_track = np.concatenate([track, ls[:, None], rs[:, None]], axis=-1)

    return new_track


def run_map_gen():
    N = 200
    path = load_track()
    path = interp_track(path, N)

    track = create_nvecs(path)
    track = set_widths(track, 5)

    # plot_race_line(track, wait=True)

    return track

def MinCurvature():
    track = run_map_gen()
    txs = track[:, 0]
    tys = track[:, 1]
    nvecs = track[:, 2:4]
    th_ns = [lib.get_bearing([0, 0], nvecs[i, 0:2]) for i in range(len(nvecs))]

    n_max = 3
    N = len(track)

    n_f_a = ca.MX.sym('n_f', N)
    n_f = ca.MX.sym('n_f', N-1)
    th_f = ca.MX.sym('n_f', N-1)

    x0_f = ca.MX.sym('x0_f', N-1)
    x1_f = ca.MX.sym('x1_f', N-1)
    y0_f = ca.MX.sym('y0_f', N-1)
    y1_f = ca.MX.sym('y1_f', N-1)
    th1_f = ca.MX.sym('y1_f', N-1)
    th2_f = ca.MX.sym('y1_f', N-1)
    th1_f1 = ca.MX.sym('y1_f', N-2)
    th2_f1 = ca.MX.sym('y1_f', N-2)

    o_x_s = ca.Function('o_x', [n_f], [track[:-1, 0] + nvecs[:-1, 0] * n_f])
    o_y_s = ca.Function('o_y', [n_f], [track[:-1, 1] + nvecs[:-1, 1] * n_f])
    o_x_e = ca.Function('o_x', [n_f], [track[1:, 0] + nvecs[1:, 0] * n_f])
    o_y_e = ca.Function('o_y', [n_f], [track[1:, 1] + nvecs[1:, 1] * n_f])

    dis = ca.Function('dis', [x0_f, x1_f, y0_f, y1_f], [ca.sqrt((x1_f-x0_f)**2 + (y1_f-y0_f)**2)])

    track_length = ca.Function('length', [n_f_a], [dis(o_x_s(n_f_a[:-1]), o_x_e(n_f_a[1:]), 
                                o_y_s(n_f_a[:-1]), o_y_e(n_f_a[1:]))])

    real = ca.Function('real', [th1_f, th2_f], [ca.cos(th1_f)*ca.cos(th2_f) + ca.sin(th1_f)*ca.sin(th2_f)])
    im = ca.Function('im', [th1_f, th2_f], [-ca.cos(th1_f)*ca.sin(th2_f) + ca.sin(th1_f)*ca.cos(th2_f)])

    sub_cmplx = ca.Function('a_cpx', [th1_f, th2_f], [ca.atan2(im(th1_f, th2_f), real(th1_f, th2_f))])
    get_th_n = ca.Function('gth', [th_f], [sub_cmplx(ca.pi*np.ones(N-1), sub_cmplx(th_f, th_ns[:-1]))])
    
    d_n = ca.Function('d_n', [n_f_a, th_f], [track_length(n_f_a)/ca.tan(get_th_n(th_f))])

    # define symbols
    n = ca.MX.sym('n', N)
    th = ca.MX.sym('th', N)


    nlp = {\
    'x': ca.vertcat(n, th),
    'f': ca.sumsqr(sub_cmplx(th[1:], th[:-1])), 
    'g': ca.vertcat(
                # dynamic constraints
                n[1:] - (n[:-1] + d_n(n, th[:-1])),

                # boundary constraints
                n[0], th[0],
                n[-1], #th[-1],
            ) \
    
    }

    S = ca.nlpsol('S', 'ipopt', nlp)

    ones = np.ones(N)
    n0 = ones*0

    th0 = []
    for i in range(N-1):
        th_00 = lib.get_bearing(track[i, 0:2], track[i+1, 0:2])
        th0.append(th_00)

    th0.append(0)
    th0 = np.array(th0)

    x0 = ca.vertcat(n0, th0)

    lbx = [-n_max] * N + [-np.pi]*N 
    ubx = [n_max] * N + [np.pi]*N 

    r = S(x0=x0, lbg=0, ubg=0, lbx=lbx, ubx=ubx)

    x_opt = r['x']

    n_set = np.array(x_opt[:N])
    thetas = np.array(x_opt[1*N:2*N])

    plot_race_line(np.array(track), n_set, wait=True)

    return n_set



if __name__ == "__main__":
    # run_map_gen()
    MinCurvature()


