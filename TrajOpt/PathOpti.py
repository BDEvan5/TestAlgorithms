import numpy as np 
import casadi as ca
from matplotlib import pyplot as plt

from TrajPlanner import *
from PathFinder import *
from RaceTrackMaps import *

    

track = np.load('TrajOpt/RaceTrack1020.npy')
# track = track / 10 #
# scale = 10m
scale = 100

env_map = RaceMap('RaceTrack1020')
s = env_map.start
e = env_map.end
path_finder = PathFinder(env_map.obs_hm._check_line, env_map.start, env_map.end)
path = path_finder.run_search(5)
# env_map.obs_hm.show_map(True, path)

xgrid = np.linspace(0, 100, 100)
ygrid = np.linspace(0, 100, 100)
# flat_track = track.ravel(order='F')
flat_track = env_map.obs_hm.race_map
flat_track = flat_track.ravel('F')
flat_track = [1 if flat_track[i] else 0 for i in range(len(flat_track))]
lut = ca.interpolant('lut', 'bspline', [xgrid, ygrid], flat_track)

print(lut([90, 48]))
print(lut([48, 90]))

N = len(path)

x = ca.MX.sym('x', N)
y = ca.MX.sym('y', N)
th = ca.MX.sym('th', N)
ds = ca.MX.sym('ds', N)

nlp = {\
    'x': ca.vertcat(x, y, th, ds),
    'f': ca.sum1(ds) ,#+ ca.sumsqr(th[1:] - th[:-1]) * 1000, # a curve term will be added here
    # 'f': ca.sumsqr(th[1:] - th[:-1]),
    'g': ca.vertcat(
        x[1:] - (x[:-1] + ds[:-1] * ca.sin(th[:-1])),
        y[1:] - (y[:-1] + ds[:-1] * ca.cos(th[:-1])),
        (ds[:-1] - (ca.sum1(ds[:-1]) / (N-2))) * 100 ,
        lut(ca.horzcat(x[:-1], y[:-1]).T).T * 10,

        x[0] - s[0], y[0] - s[1],
        x[-1] - e[0], y[-1] - e[1]
        # th[-1], ds[-1]
    )\
    }

S = ca.nlpsol('S', 'ipopt', nlp, {'ipopt':{'print_level':3, 'max_iter':1000}})

path = np.array(path)
x0 = path[:, 0]
y0 = path[:, 1]
th0, ds0 = [], []
for i in range(N-1):
    th0.append(lib.get_bearing(path[i], path[i+1]))
    ds0.append(lib.get_distance(path[i], path[i+1]))
th0.append(0)
ds0.append(0)

x00 = ca.vertcat(x0, y0, th0, ds0)

lbx = [0] * N + [0]*N + [-np.pi] * N + [0] * N
ubx = [100] * N + [100]*N + [np.pi] * N + [10] * N

n1 = N-1
lbg = [0] * n1 + [0]*n1 + [0] * n1  + [-0.1] * n1 + [0] * 4
ubg = [0] * n1 + [0]*n1 + [0] *n1 + [0.1] * n1 + [0] * 4

r = S(x0=x00, lbg=lbg, ubg=ubg, lbx=lbx, ubx=ubx)

print(f"Solution found")
x_opt = r['x']
# print(x_opt)

x = np.array(x_opt[:N])
y = np.array(x_opt[N:N*2])
th = np.array(x_opt[2*N:N*3])
ds = np.array(x_opt[3*N:N*4])

# print(x)
# print(y)
# print(th)
# print(ds)

path = np.concatenate([x, y], axis=-1)
env_map.obs_hm.show_map(True, path)
env_map.race_course.show_map(True, path)


