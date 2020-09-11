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

def smooth_track(track):
    N = len(track)
    xs = track[:, 0]
    ys = track[:, 1]

    th1_f = ca.MX.sym('y1_f', N-2)
    th2_f = ca.MX.sym('y2_f', N-2)
    x_f = ca.MX.sym('x_f', N)
    y_f = ca.MX.sym('y_f', N)

    real = ca.Function('real', [th1_f, th2_f], [ca.cos(th1_f)*ca.cos(th2_f) + ca.sin(th1_f)*ca.sin(th2_f)])
    im = ca.Function('im', [th1_f, th2_f], [-ca.cos(th1_f)*ca.sin(th2_f) + ca.sin(th1_f)*ca.cos(th2_f)])

    sub_cmplx = ca.Function('a_cpx', [th1_f, th2_f], [ca.atan(im(th1_f, th2_f)/real(th1_f, th2_f))])

    d_th = ca.Function('d_th', [x_f, y_f], [ca.if_else(ca.fabs(x_f[1:] - x_f[:-1]) < 0.01 ,ca.atan((y_f[1:] - y_f[:-1])/(x_f[1:] - x_f[:-1])), 10000)])

    x = ca.MX.sym('x', N)
    y = ca.MX.sym('y', N)
    th = ca.MX.sym('th', N-1)

    B = 5
    nlp = {\
        'x': ca.vertcat(x, y, th),
        'f': ca.sumsqr(sub_cmplx(th[1:], th[:-1])) + B* (ca.sumsqr(x-xs) + ca.sumsqr(y-ys)),
        # 'f':  B* (ca.sumsqr(x-xs) + ca.sumsqr(y-ys)),
        'g': ca.vertcat(\
                th - d_th(x, y),
                x[0] - xs[0], y[0]- ys[0],
                x[-1] - xs[-1], y[-1]- ys[-1],
            )\
        }

    S = ca.nlpsol('S', 'ipopt', nlp)
    # th0 = [lib.get_bearing(track[i, 0:2], track[i+1, 0:2]) for i in range(N-1)]
    th0 = d_th(xs, ys)

    x0 = ca.vertcat(xs, ys, th0)

    lbx = [0] *2* N + [-np.pi]*(N -1)
    ubx = [100] *2 * N + [np.pi]*(N-1) 

    r = S(x0=x0, lbg=0, ubg=0, lbx=lbx, ubx=ubx)

    print(f"Solution found")
    x_opt = r['x']

    xs_new = np.array(x_opt[:N])
    ys_new = np.array(x_opt[N:2*N])

    track[:, 0] = xs_new[:, 0]
    track[:, 1] = ys_new[:, 0]

    return track

    # plot_race_line(track, wait=True)

# def spline_track(track):
#     N = len(track)
#     xs = track[:, 0]
#     ys = track[:, 1]



def get_nvec(x0, x2):
    th = lib.get_bearing(x0, x2)
    new_th = th + np.pi/2
    nvec = lib.theta_to_xy(new_th)

    return nvec


def create_nvecs(track):
    N = len(track)

    new_track, nvecs = [], []
    new_track.append(track[0, :])
    nvecs.append(get_nvec(track[0, :], track[1, :]))
    for i in range(len(track)-1):
        pt1 = new_track[max(-1, 0)]
        pt2 = track[min((i, N)), :]
        pt3 = track[min((i+1, N-1)), :]

        th1 = lib.get_bearing(pt1, pt2)
        th2 = lib.get_bearing(pt2, pt3)
        if th1 == th2:
            pass
        else:
            th = lib.add_angles_complex(th1, th2) / 2

            new_th = th + np.pi/2
            nvec = lib.theta_to_xy(new_th)
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
    ths = [lib.get_bearing(track[i, 0:2], track[i+1, 0:2]) for i in range(N-1)]

    ls, rs = [width], [width]
    for i in range(N-2):
        dth = lib.sub_angles_complex(ths[i+1], ths[i])
        dw = dth / (np.pi) * width
        l = width #+  dw
        r = width #- dw
        ls.append(l)
        rs.append(r)

    ls.append(width)
    rs.append(width)

    ls = np.array(ls)
    rs = np.array(rs)

    new_track = np.concatenate([track, ls[:, None], rs[:, None]], axis=-1)

    return new_track

def run_map_gen():
    N = 200
    path = load_track()
    # path = interp_track(path, N)

    # track = smooth_track(path)

    track = create_nvecs(path)
    track = set_widths(track, 5)


    plot_race_line(track, wait=True)

    return track


if __name__ == "__main__":
    run_map_gen()

