import numpy as np 
from casadi import *
from matplotlib import pyplot as plt

from TrajPlanner import *

# get seed, lut, odom, A, B, T, deltaT


N = int(T/delta_t) # T / t_delta

x = MX.sym('x', N)
y = MX.sym('y', N)
theta = MX.sym('theta', N)
v = MX.sym('v', N)
w = MX.sym('w', N)

nlp = {\
    'x':vertcat(x,y,theta,v,w),
    'f': A*(sumsqr(v) + sumsqr(w)) + B*(sumsqr(x-seed[:,0]) + sumsqr(y-seed[:,1])),
    'g':vertcat(\
                x[1:] - (x[:-1] + delta_t*v[:-1]*cos(theta[:-1])),
                y[1:] - (y[:-1] + delta_t*v[:-1]*sin(theta[:-1])),
                theta[1:] - (theta[:-1] + delta_t*w[:-1]),
                x[0]-odom[0],y[0]-odom[1],theta[0]-odom[2],
#                    x[-1]-seed[-1,0],y[-1]-seed[-1,1],
                lut(horzcat(x+1.2,y+1.2).T).T # TODO: Hardcoded
                )\
}


S = nlpsol('S', 'ipopt', nlp, {'ipopt':{'print_level':3, 'max_iter':max_iter}})

x0 = initSolution(seed)

#     r = S(x0=x0, lbg=[0]*(N-1)*3+[0]*5, ubg=([0]*(N-1)*3+[0]*5), ubx=[3]*N+[3]*N+[np.inf]*N+[1]*N+[2]*N, lbx=[-3]*N+[-3]*N+[-np.inf]*N+[0]*N+[-2]*N)
r = S(x0=x0, lbg=[0]*(N-1)*3+[0]*3+[0.06]*N, ubg=([0]*(N-1)*3+[0]*3+[inf]*N), ubx=[1]*N+[1]*N+[np.inf]*N+[1]*N+[1]*N, lbx=[-1]*N+[-1]*N+[-np.inf]*N+[0]*N+[-1]*N)
#     r = S(x0=x0, lbg=[0]*(N-1)*3+[0]*5+[0.06]*N, ubg=([0]*(N-1)*3+[0]*5+[inf]*N), ubx=[1]*N+[1]*N+[np.inf]*N+[1]*N+[1]*N, lbx=[-1]*N+[-1]*N+[-np.inf]*N+[0]*N+[-1]*N)
x_opt = r['x']
#     print S.stats()

traj = np.array([\
                    np.array(x_opt[0:N]),\
                    np.array(x_opt[N:2*N]),\
                    np.array(x_opt[2*N:3*N]),\
                    np.array(x_opt[3*N:4*N]),\
                    np.array(x_opt[4*N:5*N]),\
                ]).T

return traj[0]





