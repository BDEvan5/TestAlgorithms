import numpy as np 
from casadi import *
from matplotlib import pyplot as plt

"""Cartpole example"""
# parameters
N = 20
d_max = 2
u_max = 20
m1 = 1
m2 = 0.3
g = 9.81
l = 0.5
d = 1
# hk = T/(N - 1)
# T = 2
# t = np.array([i*hk for i in range(N)])

# dynamics equations
q1_f = MX.sym('q1_f', N-1)
q2_f = MX.sym('q2_f', N-1)
q2_dot_f = MX.sym('q2_f', N-1)
u_f = MX.sym('u_f', N-1)
ddq1 = Function('ddq1', [q1_f, q2_f, q2_dot_f, u_f], [(l * m2 * sin(q2_f) * q2_dot_f**2 + u_f + m2 * g * cos(q2_f) * sin(q2_f)) / (m1 + m2 *(1-cos(q2_f)**2))])
ddq2 = Function('ddq2', [q1_f, q2_f, q2_dot_f, u_f], [-(l * m2 * cos(q2_f) *sin(q2_f) * q2_dot_f**2 + u_f *cos(q2_f) + (m1 + m2) *g * sin(q2_f)) / (l*m1 + m2 * (1-cos(q2_f)**2))])

# define symbols
q1 = MX.sym('q1', N)
q2 = MX.sym('q2', N)
q1_dot = MX.sym('q1_dot', N)
q2_dot = MX.sym('q2_dot', N)
u = MX.sym('u', N)
t = MX.sym('t', N)
t_ones = np.ones(N-1)

nlp = {\
    'x':  vertcat(q1, q2, q1_dot, q2_dot, u, t),
    'f':  t[-1],
    'g': vertcat(
                # dynamic constraints
                q1_dot[1:] - (q1_dot[:-1] + t[-1]/(N-1) / 2 *(ddq1(q1[:-1], q2[:-1], q2_dot[:-1], u[:-1]) + ddq1(q1[1:], q2[1:], q2_dot[1:], u[1:]))),
                q2_dot[1:] - (q2_dot[:-1] + t[-1]/(N-1) / 2 *(ddq2(q1[:-1], q2[:-1], q2_dot[:-1], u[:-1]) + ddq2(q1[1:], q2[1:], q2_dot[1:], u[1:]))),
                q1[1:] - (q1[:-1] + t[-1]/(N-1) /2 * (q1_dot[1:] + q1_dot[:-1])),
                q2[1:] - (q2[:-1] + t[-1]/(N-1) /2 * (q2_dot[1:] + q2_dot[:-1])),

                # boundary constraints
                q1[0] , q2[0] , q1_dot[0] , q2_dot[0],
                q1[-1] - d, q2[-1] - pi, q1_dot[-1], q2_dot[-1],

                # time constraint
                t[1:] - (t[:-1] + t[-1] /(N -1) * t_ones),
                t[0]
            )  \
    }

S = nlpsol('S', 'ipopt', nlp)

T0 = 2
hk = T0/(N - 1)
t0 = np.array([i*hk for i in range(N)])

x0 = vertcat(t0 * (d/T0), t0 * (pi/T0), t0*0, t0*0, t0*0, t0)

lbx = [-d_max]*N + [-np.inf]*N + [-np.inf]*N + [-np.inf]*N + [-u_max]*N + [0]*N
ubx = [d_max]*N + [np.inf]*N + [np.inf]*N + [np.inf]*N + [u_max]*N + [np.inf]*N

r = S(x0=x0, lbg=0, ubg=0, lbx=lbx, ubx=ubx)

print(f"Solution found")
# print(r)
x_opt = r['x']
# print('u_opt: ', x_opt)

q1_out = np.array(x_opt[0:N])
q2_out = np.array(x_opt[N:2*N])
q1_dot_out = np.array(x_opt[2*N:3*N])
q2_dot_out = np.array(x_opt[3*N:4*N])
u_out = np.array(x_opt[4*N:5*N])
t_out = np.array(x_opt[5*N:])

# plot output
plt.figure(1)
plt.plot(t_out, q1_out, 'x')
plt.plot(t_out, x0[0:N])

plt.figure(2)
plt.plot(t_out, q2_out, 'x')
plt.plot(t_out, x0[N:N*2])


plt.figure(3)
plt.plot(t_out, u_out, 'x')
plt.plot(t_out, x0[4*N:5*N])


plt.figure(4)
xs = q1_out + l * sin(q2_out)
ys = - l * cos(q2_out)
plt.plot(xs, ys, linewidth=3)
plt.plot(q1_out, np.zeros_like(q1_out), 'x')

for i in range(N):
    xs = [q1_out[i], q1_out[i] + l * sin(q2_out[i])]
    ys = [0, - l * cos(q2_out[i])]
    plt.plot(xs, ys)


plt.show()


