import numpy as np 
import casadi as ca
from matplotlib import pyplot as plt

from MapGen import run_map_gen, plot_race_line
# from TrajPlanner import plot_race_line
import LibFunctions as lib

def MinDistance():
    track = run_map_gen()
    txs = track[:, 0]
    tys = track[:, 1]
    nvecs = track[:, 2:4]

    n_max = 5
    N = len(track)

    n_f_a = ca.MX.sym('n_f', N)
    n_f = ca.MX.sym('n_f', N-1)

    x0_f = ca.MX.sym('x0_f', N-1)
    x1_f = ca.MX.sym('x1_f', N-1)
    y0_f = ca.MX.sym('y0_f', N-1)
    y1_f = ca.MX.sym('y1_f', N-1)

    o_x_s = ca.Function('o_x', [n_f], [track[:-1, 0] + nvecs[:-1, 0] * n_f])
    o_y_s = ca.Function('o_y', [n_f], [track[:-1, 1] + nvecs[:-1, 1] * n_f])
    o_x_e = ca.Function('o_x', [n_f], [track[1:, 0] + nvecs[1:, 0] * n_f])
    o_y_e = ca.Function('o_y', [n_f], [track[1:, 1] + nvecs[1:, 1] * n_f])

    dis = ca.Function('dis', [x0_f, x1_f, y0_f, y1_f], [ca.sqrt((x1_f-x0_f)**2 + (y1_f-y0_f)**2)])

    track_length = ca.Function('length', [n_f_a], [dis(o_x_s(n_f_a[:-1]), o_x_e(n_f_a[1:]), 
                                o_y_s(n_f_a[:-1]), o_y_e(n_f_a[1:]))])

    grad = ca.Function('grad', [n_f_a], [(o_y_e(n_f_a[1:]) - o_y_s(n_f_a[:-1]))/ca.fmax(o_x_e(n_f_a[1:]) - o_x_s(n_f_a[:-1]), 0.1*np.ones(N-1) )])
    curvature = ca.Function('curv', [n_f_a], [grad(n_f_a)[1:] - grad(n_f_a)[:-1]]) # changes in grad

    # define symbols
    n = ca.MX.sym('n', N)

    nlp = {\
    'x': ca.vertcat(n),
    # 'f': ca.sumsqr(track_length(n)),
    'f': ca.sumsqr(curvature(n)), 
    'g': ca.vertcat(
                # boundary constraints
                n[0], 
                n[-1], 
            ) \
    
    }

    S = ca.nlpsol('S', 'ipopt', nlp)

    ones = np.ones(N)
    n0 = ones*0

    x0 = ca.vertcat(n0)

    lbx = [-n_max] * N
    ubx = [n_max] * N

    r = S(x0=x0, lbg=0, ubg=0, lbx=lbx, ubx=ubx)

    print(f"Solution found")
    x_opt = r['x']

    n_set = np.array(x_opt[:N])

    plot_race_line(np.array(track), n_set, width=5, wait=True)
    
def MinCurvature():
    track = run_map_gen()
    txs = track[:, 0]
    tys = track[:, 1]
    nvecs = track[:, 2:4]
    th_ns = [lib.get_bearing([0, 0], nvecs[i, 0:2]) for i in range(len(nvecs))]
    sls = np.sqrt(np.sum(np.power(np.diff(track[:, :2], axis=0), 2), axis=1))


    n_max = 3
    N = len(track)

    n_f_a = ca.MX.sym('n_f', N)
    n_f = ca.MX.sym('n_f', N-1)
    th_f_a = ca.MX.sym('n_f', N)
    th_f = ca.MX.sym('n_f', N-1)

    x0_f = ca.MX.sym('x0_f', N-1)
    x1_f = ca.MX.sym('x1_f', N-1)
    y0_f = ca.MX.sym('y0_f', N-1)
    y1_f = ca.MX.sym('y1_f', N-1)

    o_x_s = ca.Function('o_x', [n_f], [track[:-1, 0] + nvecs[:-1, 0] * n_f])
    o_y_s = ca.Function('o_y', [n_f], [track[:-1, 1] + nvecs[:-1, 1] * n_f])
    o_x_e = ca.Function('o_x', [n_f], [track[1:, 0] + nvecs[1:, 0] * n_f])
    o_y_e = ca.Function('o_y', [n_f], [track[1:, 1] + nvecs[1:, 1] * n_f])

    dis = ca.Function('dis', [x0_f, x1_f, y0_f, y1_f], [ca.sqrt((x1_f-x0_f)**2 + (y1_f-y0_f)**2)])

    track_length = ca.Function('length', [n_f_a], [dis(o_x_s(n_f_a[:-1]), o_x_e(n_f_a[1:]), 
                                o_y_s(n_f_a[:-1]), o_y_e(n_f_a[1:]))])

    d_n = ca.Function('d_n', [n_f, th_f], [sls/ca.tan(th_ns[:-1] - th_f)])
    # curvature = ca.Function('curv', [th_f_a], [th_f_a[1:] - th_f_a[:-1]])
    grad = ca.Function('grad', [n_f_a], [(o_y_e(n_f_a[1:]) - o_y_s(n_f_a[:-1]))/ca.fmax(o_x_e(n_f_a[1:]) - o_x_s(n_f_a[:-1]), 0.1*np.ones(N-1) )])
    curvature = ca.Function('curv', [n_f_a], [grad(n_f_a)[1:] - grad(n_f_a)[:-1]]) # changes in grad
   

    # define symbols
    n = ca.MX.sym('n', N)
    th = ca.MX.sym('th', N)


    nlp = {\
    'x': ca.vertcat(n, th),
    'f': ca.sumsqr(curvature(n)),
    # 'f': ca.sumsqr(track_length(n)),
    'g': ca.vertcat(
                # dynamic constraints
                n[1:] - (n[:-1] + d_n(n[:-1], th[:-1])),

                # boundary constraints
                n[0], th[0],
                n[-1], th[-1],
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

    print(f"Solution found")
    x_opt = r['x']

    n_set = np.array(x_opt[:N])
    thetas = np.array(x_opt[1*N:2*N])

    plot_race_line(np.array(track), n_set, width=3, wait=True)


if __name__ == "__main__":
    # MinDistance()
    MinCurvature()