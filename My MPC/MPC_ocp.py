import numpy as np 
import casadi as ca
from matplotlib import pyplot as plt
import LibFunctions as lib


class MPC:
    def __init__(self, n_pts, max_t=20):
        self.n = n_pts
        self.max_t = max_t
        self.dt = ca.MX.sym('dt', self.n-1)

        self.states = []
        self.controls = []

        self.state_der = {}
        self.initial = {}
        self.objectives = {}
        self.o_scales = {}

        self.min_lims = {}
        self.max_lims = {}


    def state(self, name="x"):
        state = ca.MX.sym(name, self.n)
        self.states.append(state)

        return state

    def control(self, name="u"):
        control = ca.MX.sym(name, self.n-1)
        self.controls.append(control)

        return control

    def get_time(self):
        return self.dt

    def set_der(self, state, der):
        self.state_der[state] = der

    def set_inital(self, state, init_val):
        self.initial[state] = init_val

    def set_objective(self, var, objective, scale=1):
        self.objectives[var] = objective 
        self.o_scales[var] = scale

    def set_lims(self, state, state_min, state_max):
        self.min_lims[state] = state_min
        self.max_lims[state] = state_max

    def set_up_solve(self):
        x_mins = [self.min_lims[state] for state in self.states] * self.n
        x_maxs = [self.max_lims[state] for state in self.states] * self.n
        u_mins = [self.min_lims[control] for control in self.controls] * (self.n - 1)
        u_maxs = [self.max_lims[control] for control in self.controls] * (self.n - 1)

        self.lbx = x_mins + u_mins + list(np.zeros(self.n-1))
        self.ubx = x_maxs + u_maxs + list(np.ones(self.n-1) * self.max_t)     

    def solve(self, x0):
        xs = ca.vcat([state for state in self.states])
        us = ca.vcat([control for control in self.controls])

        dyns = ca.vcat([var[1:] - (var[:-1] + self.state_der[var] * self.dt) for var in self.states])
        cons = ca.vcat([state[0] - x0[i] for i, state in enumerate(self.states)])

        obs = ca.vcat([(o - self.objectives[o]) * self.o_scales[o] for o in self.objectives.keys()])

        nlp = {\
            'x': ca.vertcat(xs, us, self.dt),
            'f': ca.sumsqr(obs),
            'g': ca.vertcat(dyns, cons)
            }

        n_g = nlp['g'].shape[0]
        self.lbg = [0] * n_g
        self.ubg = [0] * n_g

        x00 = ca.vcat([self.initial[state] for state in self.states])
        u00 = ca.vcat([self.initial[control] for control in self.controls])
        x0 = ca.vertcat(x00, u00, self.initial[self.dt])

        S = ca.nlpsol('S', 'ipopt', nlp, {'ipopt':{'print_level':0}})
        r = S(x0=x0, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg)
        x_opt = r['x']

        n_state_vars = len(self.states) * self.n
        n_control_vars = len(self.controls) * self.n

        states = np.array(x_opt[:n_state_vars])
        controls = np.array(x_opt[n_state_vars:n_state_vars + n_control_vars])
        times = np.array(x_opt[-self.n+1:])

        for i, state in enumerate(self.states):
            self.set_inital(state, states[i*self.n:self.n*(i+1)])

        for i, control in enumerate(self.controls):
            self.set_inital(control, controls[(self.n-1)*i: (i+1) * (self.n-1)])

        self.set_inital(self.dt, times)

        return states, controls, times



def example():
    pathpoints = 30
    ref_path = {}
    ref_path['x'] = 5*np.sin(np.linspace(0,2*np.pi, pathpoints+1))
    ref_path['y'] = np.linspace(1,2, pathpoints+1)**2*10
    wp = ca.horzcat(ref_path['x'], ref_path['y']).T

    N = 5
    N1 = N-1

    wpts = np.array(wp[:, 0:N])

    ocp = Ocp(N)

    x = ocp.state('x')
    x_dot = ocp.control('x_dot')
    dt = ocp.get_time()

    ocp.set_der(x, x_dot)
    ocp.set_objective(x, wpts[0, :])
    ocp.set_objective(dt, ca.GenMX_zeros(N1))
    ocp.set_constraints(x, wpts[0, 0])

    x00 = wpts[0, :]
    T = 5
    dt00 = [T/N1] * N1
    xd00 = (x00[1:] - x00[:-1]) / dt00
    ocp.set_inital(x, x00)
    ocp.set_inital(x_dot, xd00)
    ocp.set_inital(dt, dt00)

    states, controls, times = ocp.solve()

    x = states
    x_dots = controls
    total_time = np.sum(times)

    print(f"Times: {times}")
    print(f"Total Time: {total_time}")
    print(f"xs: {x.T}")
    print(f"X dots: {x_dots.T}")


    plt.figure(1)
    plt.plot(wpts[0, :], np.ones_like(wpts[0, :]), 'o', markersize=12)

    plt.plot(x, np.ones_like(x), '+', markersize=20)

    plt.show()


def example2D():
    pathpoints = 30
    ref_path = {}
    ref_path['x'] = 5*np.sin(np.linspace(0,2*np.pi, pathpoints+1))
    ref_path['y'] = np.linspace(1,2, pathpoints+1)**2*10
    wp = ca.horzcat(ref_path['x'], ref_path['y']).T

    N = 10
    N1 = N-1

    wpts = np.array(wp[:, 0:N])

    ocp = MPC(N)

    x = ocp.state('x')
    y = ocp.state('y')
    x_dot = ocp.control('x_dot')
    y_dot = ocp.control('y_dot')
    dt = ocp.get_time()

    ocp.set_der(x, x_dot)
    ocp.set_der(y, y_dot)

    ocp.set_objective(x, wpts[0, :])
    ocp.set_objective(y, wpts[1, :])
    ocp.set_objective(dt, ca.GenMX_zeros(N1), 0.01)

    ocp.set_constraints(x, wpts[0, 0])
    ocp.set_constraints(y, wpts[1, 0])

    max_speed = 1
    ocp.set_lims(x, 0, ca.inf)
    ocp.set_lims(y, 0, ca.inf)
    ocp.set_lims(x_dot, -max_speed, max_speed)
    ocp.set_lims(y_dot, -max_speed, max_speed)

    T = 5
    dt00 = [T/N1] * N1
    ocp.set_inital(dt, dt00)
    x00 = wpts[0, :]
    y00 = wpts[1, :]
    xd00 = (x00[1:] - x00[:-1]) / dt00
    ocp.set_inital(x, x00)
    ocp.set_inital(x_dot, xd00)
    yd00 = (y00[1:] - y00[:-1]) / dt00
    ocp.set_inital(y, y00)
    ocp.set_inital(y_dot, yd00)

    ocp.set_up_solve()
    states, controls, times = ocp.solve([wpts[0, 0], wpts[1, 0]])

    x = states[:N]
    y = states[N:]
    x_dots = controls[:N1]
    y_dots = controls[N1:]
    total_time = np.sum(times)

    print(f"Times: {times}")
    print(f"Total Time: {total_time}")
    print(f"xs: {x.T}")
    print(f"ys: {y.T}")
    print(f"X dots: {x_dots.T}")
    print(f"Y dots: {y_dots.T}")


    plt.figure(1)
    plt.plot(wpts[0, :], wpts[1, :], 'o', markersize=12)

    plt.plot(x, y, '+', markersize=20)

    plt.show()


def example_loop():
    pathpoints = 30
    ref_path = {}
    ref_path['x'] = 5*np.sin(np.linspace(0,2*np.pi, pathpoints+1))
    ref_path['y'] = np.linspace(1,2, pathpoints+1)**2*10
    wp = ca.horzcat(ref_path['x'], ref_path['y']).T

    N = 10
    N1 = N-1

    

    ocp = MPC(N)

    x = ocp.state('x')
    y = ocp.state('y')
    x_dot = ocp.control('x_dot')
    y_dot = ocp.control('y_dot')
    dt = ocp.get_time()

    ocp.set_der(x, x_dot)
    ocp.set_der(y, y_dot)

    ocp.set_objective(dt, ca.GenMX_zeros(N1), 0.01)

    max_speed = 1
    ocp.set_lims(x, -ca.inf, ca.inf)
    ocp.set_lims(y, -ca.inf, ca.inf)
    ocp.set_lims(x_dot, -max_speed, max_speed)
    ocp.set_lims(y_dot, -max_speed, max_speed)

    wpts = np.array(wp[:,0:N])

    T = 5
    dt00 = [T/N1] * N1
    ocp.set_inital(dt, dt00)
    x00 = wpts[0, :]
    y00 = wpts[1, :]
    xd00 = (x00[1:] - x00[:-1]) / dt00
    ocp.set_inital(x, x00)
    ocp.set_inital(x_dot, xd00)
    yd00 = (y00[1:] - y00[:-1]) / dt00
    ocp.set_inital(y, y00)
    ocp.set_inital(y_dot, yd00)

    ocp.set_up_solve()

    for i in range(20):
        wpts = np.array(wp[:, i:i+N])
        x0 = [wpts[0, 0], wpts[1, 0]]

        ocp.set_objective(x, wpts[0, :])
        ocp.set_objective(y, wpts[1, :])

        states, controls, times = ocp.solve(x0)

        xs = states[:N]
        ys = states[N:]
        x_dots = controls[:N1]
        y_dots = controls[N1:]
        total_time = np.sum(times)

        print(f"Times: {times}")
        print(f"Total Time: {total_time}")
        print(f"xs: {xs.T}")
        print(f"ys: {ys.T}")
        print(f"X dots: {x_dots.T}")
        print(f"Y dots: {y_dots.T}")

        plt.figure(1)
        plt.clf()
        plt.plot(wpts[0, :], wpts[1, :], 'o', markersize=12)

        plt.plot(xs, ys, '+', markersize=20)

        plt.pause(0.5)

def example_loop_v_th():
    pathpoints = 30
    ref_path = {}
    ref_path['x'] = 5*np.sin(np.linspace(0,2*np.pi, pathpoints+1))
    ref_path['y'] = np.linspace(1,2, pathpoints+1)**2*10
    wp = ca.horzcat(ref_path['x'], ref_path['y']).T

    N = 10
    N1 = N-1

    ocp = MPC(N)

    x = ocp.state('x')
    y = ocp.state('y')

    th = ocp.control('th')
    v = ocp.control('v')

    dt = ocp.get_time()

    ocp.set_der(x, v*ca.cos(th))
    ocp.set_der(y, v*ca.sin(th))

    ocp.set_objective(dt, ca.GenMX_zeros(N1), 0.01)

    # set limits
    max_speed = 10
    ocp.set_lims(x, -ca.inf, ca.inf)
    ocp.set_lims(y, -ca.inf, ca.inf)
    ocp.set_lims(v, 0, max_speed)
    ocp.set_lims(th, -ca.pi, ca.pi)

    wpts = np.array(wp[:,0:N])

    # find starting vals
    T = 5
    dt00 = np.array([T/N1] * N1)
    ocp.set_inital(dt, dt00)
    x00 = wpts[0, :]
    ocp.set_inital(x, x00)
    y00 = wpts[1, :]
    ocp.set_inital(y, y00)
    th00 = [lib.get_bearing(wpts[:, i], wpts[:, i+1]) for i in range(N1)]
    ocp.set_inital(th, th00)
    v00 = np.array([lib.get_distance(wpts[:, i], wpts[:, i+1]) for i in range(N1)]) / dt00
    ocp.set_inital(v, v00)

    ocp.set_up_solve()

    for i in range(20):
        wpts = np.array(wp[:, i:i+N])
        x0 = [wpts[0, 0], wpts[1, 0]]

        ocp.set_objective(x, wpts[0, :])
        ocp.set_objective(y, wpts[1, :])

        states, controls, times = ocp.solve(x0)

        xs = states[:N]
        ys = states[N:]
        ths = controls[:N1]
        vs = controls[N1:]
        total_time = np.sum(times)

        print(f"Times: {times}")
        print(f"Total Time: {total_time}")
        print(f"xs: {xs.T}")
        print(f"ys: {ys.T}")
        print(f"Thetas: {ths.T}")
        print(f"Velocities: {vs.T}")

        plt.figure(1)
        plt.clf()
        plt.plot(wpts[0, :], wpts[1, :], 'o', markersize=12)

        plt.plot(xs, ys, '+', markersize=20)

        plt.pause(0.5)




if __name__ == "__main__":
    # example()
    # example2D()
    # example_loop()
    example_loop_v_th()

