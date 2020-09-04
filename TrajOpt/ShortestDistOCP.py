import numpy as np 
from casadi import *
from matplotlib import pyplot as plt

from TrajPlanner import *

# possibly simplify track
track = load_track('TrajOpt/RaceTrack1000_abscissa.csv',show=False)
# track = track/10
cs, cy, nvecs = calc_splines(track[:, 0:2])
# track = np.concatenate([track, nvecs], axis=-1)

ms = nvecs[:, 1] / nvecs[:, 0] # dy/dx
cs = track[:, 1] / (track[:, 0] * ms)

track[:, 2] = ms 
track[:, 3] = cs

# calculate nvecs and add to dim, 4, 5

l = 0.33
a_max = 7.5
d_max = 0.4
n_max = 5
N = len(track)

# dynamics eqns
n_f_a = MX.sym('n_f', N)
n_f = MX.sym('n_f', N-1)
x_f = MX.sym('x_f', N-1)
y_f = MX.sym('y_f', N-1)
d_f = MX.sym('d_f', N-1)
th_f = MX.sym('th_f', N-1)
v_f = MX.sym('v_f', N-1)

x0_f = MX.sym('x0', N-1)
x1_f = MX.sym('x1', N-1)
y0_f = MX.sym('y0', N-1)
y1_f = MX.sym('y1', N-1)

c0_f = MX.sym('c0', N-1)
c1_f = MX.sym('c1', N-1)
m0_f = MX.sym('m0', N-1)
m1_f = MX.sym('m1', N-1)

inter_x = Function('inter', [c0_f, m0_f], [(track[1:, 3] - c0_f)/(m0_f - track[1:, 2])])
inter_y = Function('inter', [c0_f, m0_f], 
                [(c0_f*track[1:, 2] - track[1:, 3]*m0_f) / (track[1:, 2] - m0_f)])

get_c = Function('get_c', [m0_f, x_f, y_f], [y_f / (m0_f * x0_f)])

get_nx = Function('nx', [n_f, th_f], [inter_x(get_c(tan(th_f), track[:-1, 0], track[:-1, 1]), tan(th_f))])
get_ny = Function('ny', [n_f, th_f], [inter_y(get_c(tan(th_f), track[:-1, 0], track[:-1, 1]), tan(th_f))])

dis = Function('dis', [x0_f, x1_f, y0_f, y1_f], [sqrt((x1_f-x0_f)**2 + (y1_f-y0_f)**2)])
d_n = Function('d_n', [n_f, th_f], [dis(get_nx(n_f, th_f), track[1:, 0], get_ny(n_f, th_f), track[1:, 1])])

d_th = Function('d_th', [v_f, d_f], [v_f/l * tan(d_f)])

o_x_s = Function('o_x', [n_f], [track[:-1, 0] + nvecs[:-1, 0] * n_f])
o_y_s = Function('o_y', [n_f], [track[:-1, 1] + nvecs[:-1, 1] * n_f])
o_x_e = Function('o_x', [n_f], [track[1:, 0] + nvecs[1:, 0] * n_f])
o_y_e = Function('o_y', [n_f], [track[1:, 1] + nvecs[1:, 1] * n_f])

track_length = Function('length', [n_f_a], [dis(o_x_s(n_f_a[:-1]), o_x_e(n_f_a[1:]), 
                            o_y_s(n_f_a[:-1]), o_y_e(n_f_a[1:]))])

# d_x = Function('d_x', [n_f, v_f, th_f, t_f], [o_x_s(n_f) + v_f * sin(th_f) * (t_f[1:] - t_f[:-1])])
# d_y = Function('d_y', [n_f, v_f, th_f, t_f], [o_y_s(n_f) + v_f * cos(th_f) ])

# d_n = Function('d_n', [n_f, th_f], [dis(d_x(n_f, th_f), o_x_e(n_f), \
                                                # d_y(n_f, th_f), o_y_e(n_f))])


# define symbols
n = MX.sym('n', N)
# t = MX.sym('t', N)
# v = MX.sym('v', N)
# th = MX.sym('th', N)
# a = MX.sym('a', N)
# d = MX.sym('d', N)

vs = np.ones(N) * 2

nlp_g = vertcat(
                # dynamic constraints
                # th[1:] - (th[:-1] + d_th(vs[:-1], d[:-1])),
                # n[1:] - (d_n(n[:-1], th[:-1])),

                # boundary constraints
                n[0], 
                n[-1],
            ) 

nlp = {\
    'x': vertcat(n),
    'f': sumsqr(track_length(n)),
    'g':  nlp_g\
    }

S = nlpsol('S', 'ipopt', nlp)

ones = np.ones(N)
n0 = ones*0

th0 = []

for i in range(N-1):
    th_00 = lib.get_bearing(track[i, 0:2], track[i+1, 0:2])
    th0.append(th_00)

th0 = np.array(0)
# calculate this afterwards
# d0 = atan(l * (th0[1:] - th0[:-1]) / vs[:-1])
# d0 = np.insert(np.array(d0), -1, 0)

x0 = vertcat(n0)

lbx = [-n_max]*N #+ [-np.pi]*N + [-d_max]*N
ubx = [n_max]*N #+[np.pi]*N + [d_max]*N

r = S(x0=x0, lbg=0, ubg=0, lbx=lbx, ubx=ubx)

print(f"Solution found")
# print(r)
x_opt = r['x']

track = load_track('TrajOpt/RaceTrack1000_abscissa.csv',show=False)
n_set = np.array(x_opt)
n_set = np.insert(n_set, 0, 0)
plot_race_line(np.array(track), n_set)





