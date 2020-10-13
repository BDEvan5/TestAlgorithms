from rockit import *
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, cos, sin, tan
from casadi import vertcat, sumsqr, symvar

ocp = Ocp(T=FreeTime(10.0))

# Bicycle model

x     = ocp.state()
y     = ocp.state()
theta = ocp.state()

delta = ocp.control()
V     = ocp.control()

L = 1
N = 20

ocp.set_der(x, V*cos(theta))
ocp.set_der(y, V*sin(theta))
ocp.set_der(theta, V/L*tan(delta))

# Initial constraints
ocp.subject_to(ocp.at_t0(x)==0)
ocp.subject_to(ocp.at_t0(y)==0)
ocp.subject_to(ocp.at_t0(theta)==pi/2)


ocp.set_initial(x,0)
ocp.set_initial(y,ocp.t)
ocp.set_initial(theta, pi/2)
ocp.set_initial(V,1)

ocp.subject_to(0 <= (V<=1))
ocp.subject_to( -pi/6 <= (delta<= pi/6))


# Define a placeholder for concrete waypoints to be defined on edges of the control grid
waypoints = ocp.parameter(2, grid='control')

# Minimal time
ocp.add_objective(ocp.T)
ocp.add_objective(10*ocp.integral((x-waypoints[0])**2+(y-waypoints[1])**2, grid='control')/ocp.T)

# Pick a solution method
ocp.solver('ipopt')

# Make it concrete for this ocp
ocp.method(MultipleShooting(N=20,M=4,intg='rk'))

# Give concerte numerical values for waypoints
waypoints_num = np.array([(i, cos(i)) for i in range(N)]).T
ocp.set_value(waypoints, waypoints_num)


# solve
sol = ocp.solve()

from pylab import *
figure()

ts, xs = sol.sample(x, grid='control')
ts, ys = sol.sample(y, grid='control')


print(xs,ys)
plot(xs, ys,'bo')
plot(waypoints_num[0,:],waypoints_num[1,:],'kx')

ts, xs = sol.sample(x, grid='integrator')
ts, ys = sol.sample(y, grid='integrator')

plot(xs, ys, 'b.')


ts, xs = sol.sample(x, grid='integrator',refine=10)
ts, ys = sol.sample(y, grid='integrator',refine=10)

plot(xs, ys, '-')

axis('equal')
show(block=True)