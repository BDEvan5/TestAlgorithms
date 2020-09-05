import numpy as np 
import casadi as ca
from matplotlib import pyplot as plt

from TrajPlanner import *

"""Simple track"""
track = np.array([[1, 1], 
                [1, 2], 
                [2, 3], 
                [2, 4], 
                [3, 5], 
                [4, 5], 
                [5, 5], 
                [6, 6], 
                [7, 6]])

# track = np.array([[1, 1], 
#                 [1, 2], 
#                 [2, 3], 
#                 [2, 4], 
#                 [3, 5], 
#                 [3, 6], 
#                 [2, 7], 
#                 [2, 8], 
#                 [1, 9]])



ws = np.ones_like(track) * 1
track = np.concatenate([track, ws], axis=-1)

# plot_track(track)
nvecs = calc_my_nvecs(track)


"""Other track"""
# track = load_track('TrajOpt/RaceTrack1000_abscissa.csv',show=False)
# track = track/10
# cs, cy, nvecs = calc_splines(track[:, 0:2])

txs = track[:, 0]
tys = track[:, 1]
ms = nvecs[:, 1] / nvecs[:, 0] # dy/dx
cs = tys / (txs * ms)

track[:, 2] = ms 
track[:, 3] = cs

l = 0.33
a_max = 7.5
d_max = 0.4
n_max = 1
N = len(track)

# dynamics eqns
n_f_a = ca.MX.sym('n_f', N)
n_f = ca.MX.sym('n_f', N-1)
x_f = ca.MX.sym('x_f', N-1)
y_f = ca.MX.sym('y_f', N-1)
d_f = ca.MX.sym('d_f', N-1)
th_f = ca.MX.sym('th_f', N-1)
v_f = ca.MX.sym('v_f', N-1)

x0_f = ca.MX.sym('x0_f', N-1)
x1_f = ca.MX.sym('x1_f', N-1)
y0_f = ca.MX.sym('y0_f', N-1)
y1_f = ca.MX.sym('y1_f', N-1)

c0_f = ca.MX.sym('c0', N-1)
c1_f = ca.MX.sym('c1', N-1)
m0_f = ca.MX.sym('m0', N-1)
m1_f = ca.MX.sym('m1', N-1)

o_x_s = ca.Function('o_x', [n_f], [track[:-1, 0] + nvecs[:-1, 0] * n_f])
o_y_s = ca.Function('o_y', [n_f], [track[:-1, 1] + nvecs[:-1, 1] * n_f])
o_x_e = ca.Function('o_x', [n_f], [track[1:, 0] + nvecs[1:, 0] * n_f])
o_y_e = ca.Function('o_y', [n_f], [track[1:, 1] + nvecs[1:, 1] * n_f])

get_c = ca.Function('get_c', [m0_f, x_f, y_f], [y_f / (m0_f * x_f)])
dis = ca.Function('dis', [x0_f, x1_f, y0_f, y1_f], [ca.sqrt((x1_f-x0_f)**2 + (y1_f-y0_f)**2)])

inter_x = ca.Function('inter', [c0_f, m0_f], [(cs[1:] - c0_f)/(m0_f - ms[1:])])
inter_y = ca.Function('inter', [c0_f, m0_f], [(c0_f*ms[1:] - cs[1:]*m0_f) / (ms[1:] - m0_f)])

get_nx = ca.Function('nx', [n_f, th_f], [inter_x(get_c(ca.tan(th_f), o_x_s(n_f), o_y_s(n_f)), ca.tan(th_f))])
get_ny = ca.Function('ny', [n_f, th_f], [inter_y(get_c(ca.tan(th_f), o_x_s(n_f), o_y_s(n_f)), ca.tan(th_f))])


d_n = ca.Function('d_n', [n_f, th_f], [dis(get_nx(n_f, th_f), track[1:, 0], get_ny(n_f, th_f), track[1:, 1])])

track_length = ca.Function('length', [n_f_a], [dis(o_x_s(n_f_a[:-1]), o_x_e(n_f_a[1:]), 
                            o_y_s(n_f_a[:-1]), o_y_e(n_f_a[1:]))])

# d_th = ca.Function('d_th', [v_f, d_f], [v_f/l * tan(d_f)])



ones = np.ones(N)
# n0 = ones*0
n0 = np.random.uniform(-1, 1, N)

th0 = []

for i in range(N-1):
    th_00 = lib.get_bearing(track[i, 0:2], track[i+1, 0:2])
    th0.append(th_00)

th0.append(0)
th0 = np.array(th0)

# arg = ca.vertcat(n0[:-1], th0[:-1])
# print(g_fcn(arg))



# define symbols
n = ca.MX.sym('n', N)
# t = ca.MX.sym('t', N)
# v = ca.MX.sym('v', N)
th = ca.MX.sym('th', N)
# a = ca.MX.sym('a', N)
# d = ca.MX.sym('d', N)



# nlp = {\
#     'x': ca.vertcat(n, th),
#     'f': ca.sumsqr(track_length(n)),
#     'g': ca.vertcat(
#                 # dynamic constraints
#                 # th[1:] - (th[:-1] + d_th(vs[:-1], d[:-1])),
#                 n[1:] - (d_n(n[:-1], th[:-1])),
#                 # n[1:] - g_fcn(ca.vertcat(n[:-1], th[:-1])),

#                 # boundary constraints
#                 n[0], th[0],
#                 n[-1], th[-1]
#             ) \
    
#     }
    
nlp = {\
    'x': ca.vertcat(n),
    'f': ca.sumsqr(track_length(n)),
    'g': ca.vertcat(
                # dynamic constraints
                # th[1:] - (th[:-1] + d_th(vs[:-1], d[:-1])),
                # n[1:] - (d_n(n[:-1], th[:-1])),

                # boundary constraints
                n[0], 
                n[-1]
            ) \
    }

S = ca.nlpsol('S', 'ipopt', nlp)


# calculate this afterwards
# d0 = atan(l * (th0[1:] - th0[:-1]) / vs[:-1])
# d0 = np.insert(np.array(d0), -1, 0)

# x0 = ca.vertcat(n0, th0)
x0 = ca.vertcat(n0)

lbx = [-n_max]*N #+ [-np.pi]*N 
ubx = [n_max]*N #+[np.pi]*N 

r = S(x0=x0, lbg=0, ubg=0, lbx=lbx, ubx=ubx)

print(f"Solution found")
# print(r)
x_opt = r['x']

# track = load_track('TrajOpt/RaceTrack1000_abscissa.csv',show=False)
track[:, 2:4] = np.ones_like(track[:, 2:4])
n_set = np.array(x_opt[:N])
n_set = np.insert(n_set, 0, 0)
plot_race_line(np.array(track), n_set)




