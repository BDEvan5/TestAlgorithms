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


ws = np.ones_like(track) * 1
track = np.concatenate([track, ws], axis=-1)

track = fix_up_track(track, 5)

# plot_track(track, True)
plot_spline_track(track, True)


"""Other track"""
# track = load_track('TrajOpt/RaceTrack1000_abscissa.csv',show=False)
# track = track/10
# # cs, cy, nvecs = calc_splines(track[:, 0:2])

nvecs = calc_my_nvecs(track)
txs = track[:, 0]
tys = track[:, 1]

th_ns = [lib.get_bearing([0, 0], nvecs[i, 0:2]) for i in range(len(nvecs))]
sls = np.sqrt(np.sum(np.power(np.diff(track[:, :2], axis=0), 2), axis=1))



l = 0.33
a_max = 7.5
d_max = 0.4
v_max = 6
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

o_x_s = ca.Function('o_x', [n_f], [track[:-1, 0] + nvecs[:-1, 0] * n_f])
o_y_s = ca.Function('o_y', [n_f], [track[:-1, 1] + nvecs[:-1, 1] * n_f])
o_x_e = ca.Function('o_x', [n_f], [track[1:, 0] + nvecs[1:, 0] * n_f])
o_y_e = ca.Function('o_y', [n_f], [track[1:, 1] + nvecs[1:, 1] * n_f])

dis = ca.Function('dis', [x0_f, x1_f, y0_f, y1_f], [ca.sqrt((x1_f-x0_f)**2 + (y1_f-y0_f)**2)])

d_n = ca.Function('d_n', [n_f, th_f], [sls/ca.tan(th_ns[:-1] - th_f)])

track_length = ca.Function('length', [n_f_a], [dis(o_x_s(n_f_a[:-1]), o_x_e(n_f_a[1:]), 
                            o_y_s(n_f_a[:-1]), o_y_e(n_f_a[1:]))])

d_th = ca.Function('d_th', [v_f, d_f], [v_f/l * ca.tan(d_f)])

# d_t = ca.Function('d_t', [n_f_a, v_f], [])
d_t = ca.Function('d_t', [n_f_a, v_f], [dis(o_x_e(n_f_a[1:]), o_x_s(n_f_a[:-1]), o_y_e(n_f_a[1:]), o_y_s(n_f_a[:-1]))/v_f ])
# d_t = ca.Function('d_t', [n_f_a, v_f], [ca.sqrt(ca.sum1(ca.power(ca.diff(, axis=0), 2), axis=1))])

ones = np.ones(N)
# vs = ones * 2


# define symbols
n = ca.MX.sym('n', N)
dt = ca.MX.sym('t', N)
v = ca.MX.sym('v', N)
th = ca.MX.sym('th', N)
a = ca.MX.sym('a', N)
d = ca.MX.sym('d', N)



nlp = {\
    'x': ca.vertcat(n, dt, v, th, a, d),
    # 'f': ca.sumsqr(track_length(n)),
    'f': ca.sum1(dt),
    'g': ca.vertcat(
                # dynamic constraints
                n[1:] - (n[:-1] + d_n(n[:-1], th[:-1])),
                th[1:] - (th[:-1] + d_th(v[:-1], d[:-1]) * (dt[:-1])),
                v[1:] - (v[:-1] + a[:-1] * dt[:-1]),
                # t[1:] - (t[:-1] + d_t(n, v[:-1])),
                dt[:-1] - (d_t(n, v[:-1])),

                # boundary constraints
                n[0], d[0], v[0] - 1, 
                n[-1], th[-1], d[-1], dt[-1]
            ) \
    
    }


S = ca.nlpsol('S', 'ipopt', nlp)


n0 = ones*0
# n0 = np.random.uniform(-1, 1, N)


th0 = []

for i in range(N-1):
    th_00 = lib.get_bearing(track[i, 0:2], track[i+1, 0:2])
    th0.append(th_00)

th0.append(0)
th0 = np.array(th0)

# calculate this afterwards
d0 = ca.atan(l * (th0[1:] - th0[:-1]) / (ones * 2)[:-1])
d0 = np.insert(np.array(d0), -1, 0)

v0, a0 = [1], []
for i in range(N-1):
    if v0[-1] < 3:
        a0.append(0.2)
    else:
        a0.append(0)
    new_v = np.sqrt(v0[-1]**2 + 2 *a0[-1] * sls[i])
    v0.append(new_v)
a0.append(0) # last a

t0 = [0]
for i in range(N-1):
    new_t =  sls[i] / v0[i] # t0[-1] 
    t0.append(new_t)


x0 = ca.vertcat(n0, t0, v0, th0, a0, d0)

lbx = [-n_max]*N + [0] * N + [0] * N + [-np.pi]*N + [-a_max] *N + [-d_max] * N
ubx = [n_max]*N + [np.inf] * N + [v_max] * N + [np.pi]*N + [a_max] *N + [d_max] * N


r = S(x0=x0, lbg=0, ubg=0, lbx=lbx, ubx=ubx)

print(f"Solution found")
x_opt = r['x']
# print(x_opt)
# track[:, 2:4] = np.ones_like(track[:, 2:4])

n_set = np.array(x_opt[:N])
times = np.array(x_opt[N:2*N])
velocities = np.array(x_opt[2*N:3*N])
thetas = np.array(x_opt[3*N:4*N])
accs = np.array(x_opt[4*N:5*N])
deltas = np.array(x_opt[5*N:6*N])


# print(f"Ns")
# print(n_set)

print(f"ts")
print(times)
# print(f"velocities")
# print(velocities)
# print("thetas")
# print(thetas)
# print("Accs")
# print(accs)
# print(f"Deltas")
# print(deltas)

plot_race_line(np.array(track), n_set)
plt.title("Race line")

plt.figure(2)
plt.title('Velocities and As')
plt.plot(np.cumsum(times), velocities)
plt.plot(np.cumsum(times), accs)

plt.pause(0.001)

plt.figure(3)
plt.title('Thetas')
plt.plot(np.cumsum(times), thetas)

plt.pause(0.0001)

# plt.figure(4)
plt.title('Deltas')
plt.plot(np.cumsum(times), deltas)

plt.pause(0.001)

# plt.figure(5)
# plt.title('Times')
# plt.plot(times, np.cumsum(times))

plt.show()