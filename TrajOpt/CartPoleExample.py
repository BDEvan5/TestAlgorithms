import numpy as np 
from casadi import *
from matplotlib import pyplot as plt

"""Casadi example"""
# x = SX.sym('x'); y = SX.sym('y'); z = SX.sym('z')
# nlp = {'x':vertcat(x,y,z), 'f':x**2+100*z**2, 'g':z+(1-x)**2-y}
# S = nlpsol('S', 'ipopt', nlp)
# print(S)


# r = S(x0=[2.5,3.0,0.75],\
#       lbg=0, ubg=0)
# x_opt = r['x']
# print('x_opt: ', x_opt)


"""Cartpole example"""
# parameters
N = 10
d_max = 2
u_max = 20
T = 2
m1 = 1
m2 = 0.3
g = 9.81
l = 0.5
d = 1
hk = T/(N - 1)
t = np.array([i*hk for i in range(N)])

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

nlp = {\
    'x':  vertcat(q1, q2, q1_dot, q2_dot, u),
    'f':  hk / 2 * (sumsqr(u[:-1]) + sumsqr(u[1:])),
    'g': vertcat(
                # dynamic constraints
                q1_dot[1:] - (q1_dot[:-1] + hk / 2 *(ddq1(q1[:-1], q2[:-1], q2_dot[:-1], u[:-1]) + ddq1(q1[1:], q2[1:], q2_dot[1:], u[1:]))),
                q2_dot[1:] - (q2_dot[:-1] + hk / 2 *(ddq2(q1[:-1], q2[:-1], q2_dot[:-1], u[:-1]) + ddq2(q1[1:], q2[1:], q2_dot[1:], u[1:]))),
                q1[1:] - (q1[:-1] + hk /2 * (q1_dot[1:] + q1_dot[:-1])),
                q2[1:] - (q2[:-1] + hk /2 * (q2_dot[1:] + q2_dot[:-1])),

                # boundary constraints
                q1[0] , q2[0] , q1_dot[0] , q2_dot[0],
                q1[-1] - d, q2[-1] - pi, q1_dot[-1], q2_dot[-1]
            )  \
    }

S = nlpsol('S', 'ipopt', nlp)

x0 = vertcat(t * (d/T), t * (pi/T), t*0, t*0, t*0)

lbx = [-d_max]*N + [-np.inf]*N + [-np.inf]*N + [-np.inf]*N + [-u_max]*N 
ubx = [d_max]*N + [np.inf]*N + [np.inf]*N + [np.inf]*N + [u_max]*N 

r = S(x0=x0, lbg=0, ubg=0, lbx=lbx, ubx=ubx)

print(f"Solution found")
# print(r)
x_opt = r['x']
print('u_opt: ', x_opt)

q1_out = np.array(x_opt[0:N])
q2_out = np.array(x_opt[N:2*N])
q1_dot_out = np.array(x_opt[2*N:3*N])
q2_dot_out = np.array(x_opt[3*N:4*N])
u_out = np.array(x_opt[4*N:])

# plot output
plt.figure(1)
plt.plot(t, q1_out, 'x')
plt.plot(t, x0[0:N])

plt.figure(2)
plt.plot(t, q2_out, 'x')
plt.plot(t, x0[N:N*2])


plt.figure(3)
plt.plot(t, u_out, 'x')
plt.plot(t, x0[4*N:])


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






