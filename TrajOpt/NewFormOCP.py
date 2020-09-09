import numpy as np 
import casadi as ca
from matplotlib import pyplot as plt

from MapGen import run_map_gen, plot_race_line
# from TrajPlanner import plot_race_line


def MinDistance():
    track = run_map_gen()
    txs = track[:, 0]
    tys = track[:, 1]
    nvecs = track[:, 2:4]

    l = 0.33
    a_max = 7.5
    d_max = 0.4
    v_max = 6
    n_max = 5
    N = len(track)

    n_f_a = ca.MX.sym('n_f', N)
    n_f = ca.MX.sym('n_f', N-1)
    th_f = ca.MX.sym('th_f', N-1)

    x0_f = ca.MX.sym('x0_f', N-1)
    x1_f = ca.MX.sym('x1_f', N-1)
    y0_f = ca.MX.sym('y0_f', N-1)
    y1_f = ca.MX.sym('y1_f', N-1)

    o_x_s = ca.Function('o_x', [n_f], [track[:-1, 0] + nvecs[:-1, 0] * n_f])
    o_y_s = ca.Function('o_y', [n_f], [track[:-1, 1] + nvecs[:-1, 1] * n_f])
    o_x_e = ca.Function('o_x', [n_f], [track[1:, 0] + nvecs[1:, 0] * n_f])
    o_y_e = ca.Function('o_y', [n_f], [track[1:, 1] + nvecs[1:, 1] * n_f])

    dis = ca.Function('dis', [x0_f, x1_f, y0_f, y1_f], [ca.sqrt((x1_f-x0_f)**2 + (y1_f-y0_f)**2)])

    # d_n = ca.Function('d_n', [n_f, th_f], [sls/ca.tan(th_ns[:-1] - th_f)])

    track_length = ca.Function('length', [n_f_a], [dis(o_x_s(n_f_a[:-1]), o_x_e(n_f_a[1:]), 
                                o_y_s(n_f_a[:-1]), o_y_e(n_f_a[1:]))])


    # define symbols
    n = ca.MX.sym('n', N)

    nlp = {\
    'x': ca.vertcat(n),
    'f': ca.sumsqr(track_length(n)),
    'g': ca.vertcat(
                # dynamic constraints
                # n[1:] - (n[:-1] + d_n(n[:-1], th[:-1])),

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


if __name__ == "__main__":
    MinDistance()