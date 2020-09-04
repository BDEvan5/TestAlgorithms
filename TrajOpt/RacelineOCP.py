import numpy as np 
from casadi import *
from matplotlib import pyplot as plt

from TrajPlanner import *

# possibly simplify track
track = load_track('TrajOpt/RaceTrack1000_abscissa.csv',show=False)
track = track/10
cs, cy, nvecs = calc_splines(track[:, 0:2])
track = np.concatenate([track, nvecs], axis=-1)
# calculate nvecs and add to dim, 4, 5


l = 0.33
a_max = 7.5
d_max = 0.4
n_max = 5
N = len(track)

# dynamics eqns
t_f = MX.sym('t_f', N)
n_f_a = MX.sym('n_f', N)
n_f = MX.sym('n_f', N-1)
x_f = MX.sym('x_f', N-1)
y_f = MX.sym('y_f', N-1)
v_f = MX.sym('v_f', N-1)
d_f = MX.sym('d_f', N-1)
th_f = MX.sym('th_f', N-1)

x0_f = MX.sym('x0', N-1)
x1_f = MX.sym('x1', N-1)
y0_f = MX.sym('y0', N-1)
y1_f = MX.sym('y1', N-1)

d_th = Function('d_th', [v_f, d_f], [v_f/l * tan(d_f)])
dis = Function('dis', [x0_f, x1_f, y0_f, y1_f], [sqrt((x1_f-x0_f)**2 + (y1_f-y0_f)**2)])

o_x_s = Function('o_x', [n_f], [track[:-1, 0] + track[:-1, 4] * n_f])
o_y_s = Function('o_y', [n_f], [track[:-1, 1] + track[:-1, 5] * n_f])
o_x_e = Function('o_x', [n_f], [track[1:, 0] + track[1:, 4] * n_f])
o_y_e = Function('o_y', [n_f], [track[1:, 1] + track[1:, 5] * n_f])

track_length = Function('length', [n_f_a], [dis(o_x_s(n_f_a[:-1]), o_x_e(n_f_a[1:]), 
                            o_y_s(n_f_a[:-1]), o_y_e(n_f_a[1:]))])

d_x = Function('d_x', [n_f, v_f, th_f, t_f], [o_x_s(n_f) + v_f * sin(th_f) * (t_f[1:] - t_f[:-1])])
d_y = Function('d_y', [n_f, v_f, th_f, t_f], [o_y_s(n_f) + v_f * cos(th_f) * (t_f[1:] - t_f[:-1])])

d_n = Function('d_n', [n_f, v_f, th_f, t_f], [dis(d_x(n_f, v_f, th_f, t_f), o_x_e(n_f), \
                                                d_y(n_f, v_f, th_f, t_f), o_y_e(n_f))])

d_t = Function('d_t', [n_f_a, v_f], [dis(o_x_e(n_f_a[1:]), o_x_s(n_f_a[:-1]), o_y_e(n_f_a[1:]), o_y_s(n_f_a[:-1]))/v_f ])

# define symbols
n = MX.sym('n', N)
t = MX.sym('t', N)
v = MX.sym('v', N)
th = MX.sym('th', N)
a = MX.sym('a', N)
d = MX.sym('d', N)

nlp = {\
    'x':  vertcat(n, t, v, th, a, d),
    # 'f':  t[-1],
    'f': sumsqr(track_length(n)),
    'g': vertcat(
                # dynamic constraints
                v[1:] - (v[:-1] + (t[1:] - t[:-1]) * a[:-1]),
                th[1:] - (th[:-1] + d_th(v[:-1], d[:-1])),
                n[1:] - (n[:-1] + (t[1:] - t[:-1]) * d_n(n[:-1], v[:-1], th[:-1], t)),
                t[1:] - (t[:-1] + d_t(n, v[:-1])),

                # boundary constraints
                n[0], t[0], v[0]-1,
                n[-1]

            )  \
    }

S = nlpsol('S', 'ipopt', nlp)


ones = np.ones(N)
n0 = ones*0

t0 = [0]
a0 = []
th0 = []
d0 = []
v0 = [1]
for i in range(N-1):
    if v0[-1] < 2:
        a0.append(0.2)
    else:
        a0.append(0)
    th_00 = lib.get_bearing(track[i, 0:2], track[i+1, 0:2])
    th0.append(th_00)
    dt_00 = lib.get_distance(track[i, 0:2], track[i+1, 0:2]) / v0[-1]
    t0.append(t0[-1] + dt_00)
    v0.append(v0[-1] + a0[-1]*dt_00)

th0.append(0)
a0.append(0)
th0 = np.array(th0)
v0 = np.array(v0)
# calculate this afterwards
d0 = atan(l * (th0[1:] - th0[:-1]) / v0[:-1])
d0 = np.insert(np.array(d0), -1, 0)


x0 = vertcat(n0, t0, v0, th0, a0, d0)

lbx = [-n_max]*N + [0]*N + [0]*N + [-np.pi]*N + [-a_max]*N + [-d_max]*N
ubx = [n_max]*N + [np.inf]*N + [np.inf]*N + [np.pi]*N + [a_max]*N + [d_max]*N

r = S(x0=x0, lbg=0, ubg=0, lbx=lbx, ubx=ubx)

print(f"Solution found")
# print(r)
x_opt = r['x']





