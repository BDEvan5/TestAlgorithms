from rockit import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from numpy import pi, cos, sin, tan, square
from casadi import vertcat, horzcat, sumsqr



# -------------------------------
# Define some functions to match current position with reference path
# -------------------------------

# Find closest point on the reference path compared witch current position
def find_closest_point(pose, reference_path, start_index):
    # x and y distance from current position (pose) to every point in
    # the reference path starting at a certain starting index
    xlist = reference_path['x'][start_index:] - pose[0]
    ylist = reference_path['y'][start_index:] - pose[1]
    # Index of closest point by Pythagoras theorem
    index_closest = start_index+np.argmin(np.sqrt(xlist*xlist + ylist*ylist))
    print('find_closest_point results in', index_closest)
    return index_closest

# Return the point on the reference path that is located at a certain distance
# from the current position
def index_last_point_fun(start_index, wp, dist):
    pathpoints = wp.shape[1]
    # Cumulative distance covered
    cum_dist = 0
    # Start looping the index from start_index to end
    for i in range(start_index, pathpoints-1):
        # Update comulative distance covered
        cum_dist += np.linalg.norm(wp[:,i] - wp[:,i+1])
        # Are we there yet?
        if cum_dist >= dist:
            return i + 1
    # Desired distance was never covered, -1 for zero-based index
    return pathpoints - 1

# Create a list of N waypoints
def get_current_waypoints(start_index, wp, N, dist):
    # Determine index at reference path that is dist away from starting point
    last_index = index_last_point_fun(start_index, wp, dist)
    # Calculate amount of indices between last and start point
    delta_index = last_index - start_index
    # Dependent on the amount of indices, do
    if delta_index >= N:
        # There are more than N path points available, so take the first N ones
        index_list = list(range(start_index, start_index+N+1))
        print('index list with >= N points:', index_list)
    else:
        # There are less than N path points available, so add the final one multiple times
        index_list = list(range(start_index, last_index)) + [last_index]*(N-delta_index+1)
        print('index list with < N points:', index_list)
    return wp[:,index_list]


# -------------------------------
# Problem parameters
# -------------------------------

Nsim    = 30            # how much samples to simulate
L       = 1             # bicycle model length
nx      = 3             # the system is composed of 3 states
nu      = 2             # the system has 2 control inputs
N       = 10            # number of control intervals

class RockitMPC:
    def __init__(self) -> None:
        self.ocp = Ocp(T=FreeTime(10.0))

        # Define states
        self.x     = self.ocp.state()
        self.y     = self.ocp.state()
        self.theta = self.ocp.state()
        self.X = vertcat(self.x, self.y, self.theta)

        # Defince controls
        self.delta = self.ocp.control()
        self.V     = self.ocp.control(order=0)

        # Define physical path parameter
        self.waypoints = self.ocp.parameter(2, grid='control')
        self.waypoint_last = self.ocp.parameter(2)
        self.p = vertcat(self.x, self.y)

        # Define parameter
        self.X_0 = self.ocp.parameter(nx)

    def init_mpc(self):
        # Specify ODE
        self.ocp.set_der(self.x,      self.V*cos(self.theta))
        self.ocp.set_der(self.y,      self.V*sin(self.theta))
        self.ocp.set_der(self.theta,  self.V/L*tan(self.delta))

        # Initial constraints
        self.ocp.subject_to(self.ocp.at_t0(self.X) == self.X_0)

        # Initial guess
        self.ocp.set_initial(self.x,      0)
        self.ocp.set_initial(self.y,      0)
        self.ocp.set_initial(self.theta,  0)

        self.ocp.set_initial(self.V,    0.5)

        # Path constraints
        max_v = 5
        self.ocp.subject_to( 0 <= (self.V <= max_v) )
        #ocp.subject_to( -0.3 <= (ocp.der(V) <= 0.3) )
        self.ocp.subject_to( -pi/6 <= (self.delta <= pi/6) )

        # Minimal time
        self.ocp.add_objective(0.50*self.ocp.T)

        self.ocp.add_objective(self.ocp.sum(sumsqr(self.p-self.waypoints), grid='control'))
        self.ocp.add_objective(sumsqr(self.ocp.at_tf(self.p)-self.waypoint_last))

        # Pick a solution method
        options = {"ipopt": {"print_level": 0}}
        options["expand"] = True
        options["print_time"] = False
        self.ocp.solver('ipopt', options)

        # Make it concrete for this ocp
        self.ocp.method(MultipleShooting(N=N, M=1, intg='rk', grid=FreeGrid(min=0.05, max=2)))

    def set_ocp_values(self, current_waypoints):
        self.ocp.set_value(self.waypoints,current_waypoints[:,:-1])
        self.ocp.set_value(self.waypoint_last,current_waypoints[:,-1])

    def set_x0(self, current_X):
        self.ocp.set_value(self.X_0, current_X)

    def solve(self):
        sol = self.ocp.solve()

        t_sol, x_sol     = sol.sample(self.x,     grid='control')
        t_sol, y_sol     = sol.sample(self.y,     grid='control')
        t_sol, theta_sol = sol.sample(self.theta, grid='control')
        t_sol, delta_sol = sol.sample(self.delta, grid='control')
        t_sol, V_sol     = sol.sample(self.V,     grid='control')

        err = sol.value(self.ocp.objective)

        self.ocp.set_initial(self.x, x_sol)
        self.ocp.set_initial(self.y, y_sol)
        self.ocp.set_initial(self.theta, theta_sol)
        self.ocp.set_initial(self.delta, delta_sol)
        self.ocp.set_initial(self.V, V_sol)

        return t_sol, x_sol, y_sol, theta_sol, delta_sol, V_sol, err

# Define reference path
pathpoints = 30
ref_path = {}
ref_path['x'] = 5*sin(np.linspace(0,2*pi, pathpoints+1))
ref_path['y'] = np.linspace(1,2, pathpoints+1)**2*10
wp = horzcat(ref_path['x'], ref_path['y']).T


# -------------------------------
# Logging variables
# -------------------------------

time_hist      = np.zeros((Nsim+1, N+1))
x_hist         = np.zeros((Nsim+1, N+1))
y_hist         = np.zeros((Nsim+1, N+1))
theta_hist     = np.zeros((Nsim+1, N+1))
delta_hist     = np.zeros((Nsim+1, N+1))
V_hist         = np.zeros((Nsim+1, N+1))

tracking_error = np.zeros((Nsim+1, 1))


# -------------------------------
# Solve the OCP wrt a parameter value (for the first time)
# -------------------------------


# First waypoint is current position
index_closest_point = 0

# Create a list of N waypoints
current_waypoints = get_current_waypoints(index_closest_point, wp, N, dist=6)

mpc = RockitMPC()
mpc.init_mpc()
mpc.set_ocp_values(current_waypoints)

# Set initial value for states
current_X = vertcat(ref_path['x'][0], ref_path['y'][0], 0)
mpc.set_x0(current_X)

# Solve the optimization problem
t_sol, x_sol, y_sol, theta_sol, delta_sol, V_sol, err = mpc.solve()

# Get discretised dynamics as CasADi function to simulate the system
Sim_system_dyn = mpc.ocp._method.discrete_system(mpc.ocp)


# Log data for post-processing
time_hist[0,:]    = t_sol
x_hist[0,:]       = x_sol
y_hist[0,:]       = y_sol
theta_hist[0,:]   = theta_sol
delta_hist[0,:]   = delta_sol
V_hist[0,:]       = V_sol

tracking_error[0] = err

# -------------------------------
# Simulate the MPC solving the OCP (with the updated state) several times
# -------------------------------

for i in range(Nsim):
    print("timestep", i+1, "of", Nsim)
    current_U = vertcat(delta_sol[0], V_sol[0])

    current_X = Sim_system_dyn(x0=current_X, u=current_U, T=t_sol[1]-t_sol[0])["xf"]

    mpc.set_x0(current_X)

    index_closest_point = find_closest_point(current_X[:2], ref_path, index_closest_point)
    current_waypoints = get_current_waypoints(index_closest_point, wp, N, dist=6)

    mpc.set_ocp_values(current_waypoints)

    t_sol, x_sol, y_sol, theta_sol, delta_sol, V_sol, err = mpc.solve()

    time_hist[i+1,:]    = t_sol
    x_hist[i+1,:]       = x_sol
    y_hist[i+1,:]       = y_sol
    theta_hist[i+1,:]   = theta_sol
    delta_hist[i+1,:]   = delta_sol
    V_hist[i+1,:]       = V_sol

    tracking_error[i+1] = err
    print('Tracking error f', tracking_error[i+1])



# -------------------------------
# Plot the results
# -------------------------------

T_start = 0
T_end = sum(time_hist[k,1] - time_hist[k,0] for k in range(Nsim+1))

fig = plt.figure()

ax2 = plt.subplot(2, 2, 1)
ax3 = plt.subplot(2, 2, 2)
ax4 = plt.subplot(2, 2, 3)
ax5 = plt.subplot(2, 2, 4)

ax2.plot(wp[0,:], wp[1,:], 'ko')
ax2.set_xlabel('x pos [m]')
ax2.set_ylabel('y pos [m]')
ax2.set_aspect('equal', 'box')

ax3.set_xlabel('T [s]')
ax3.set_ylabel('pos [m]')
ax3.set_xlim(0,T_end)

ax4.axhline(y= pi/6, color='r')
ax4.axhline(y=-pi/6, color='r')
ax4.set_xlabel('T [s]')
ax4.set_ylabel('delta [rad/s]')
ax4.set_xlim(0,T_end)

ax5.axhline(y=0, color='r')
ax5.axhline(y=1, color='r')
ax5.set_xlabel('T [s]')
ax5.set_ylabel('V [m/s]')
ax5.set_xlim(0,T_end)

# fig2 = plt.figure()
# ax6 = plt.subplot(1,1,1)

for k in range(Nsim+1):
    # ax6.plot(time_hist[k,:], delta_hist[k,:])
    # ax6.axhline(y= pi/6, color='r')
    # ax6.axhline(y=-pi/6, color='r')

    ax2.plot(x_hist[k,:], y_hist[k,:], 'b-')
    ax2.plot(x_hist[k,:], y_hist[k,:], 'g.')
    ax2.plot(x_hist[k,0], y_hist[k,0], 'ro', markersize = 10)

    ax3.plot(T_start, x_hist[k,0], 'b.')
    ax3.plot(T_start, y_hist[k,0], 'r.')

    ax4.plot(T_start, delta_hist[k,0], 'b.')
    ax5.plot(T_start, V_hist[k,0],     'b.')

    T_start = T_start + (time_hist[k,1] - time_hist[k,0])
    plt.pause(0.05)

ax3.legend(['x pos [m]','y pos [m]'])

fig3 = plt.figure()
ax1 = plt.subplot(1,1,1)
ax1.semilogy(tracking_error)
ax1.set_xlabel('N [-]')
ax1.set_ylabel('obj f')

plt.show(block=True)




