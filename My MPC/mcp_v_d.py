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


    def state(self, name="x", size=None):
        if size is None:
            size = self.n
        state = ca.MX.sym(name, size)
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

        # S = ca.nlpsol('S', 'ipopt', nlp, {'ipopt':{'print_level':0}})
        S = ca.nlpsol('S', 'ipopt', nlp, {'ipopt':{'print_level':5}})
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


def example_loop_v_d():
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
    th = ocp.state('th')

    d = ocp.control('d')
    v = ocp.control('v')

    dt = ocp.get_time()

    L = 1
    ocp.set_der(x, v*ca.cos(th[:-1]))
    ocp.set_der(y, v*ca.sin(th[:-1]))
    ocp.set_der(th, v/L * ca.tan(d))

    # ocp.set_objective(dt, ca.GenMX_zeros(N1), 0.01)

    # set limits
    max_speed = 10
    ocp.set_lims(x, -ca.inf, ca.inf)
    ocp.set_lims(y, -ca.inf, ca.inf)
    ocp.set_lims(v, 0, max_speed)
    ocp.set_lims(th, -ca.pi, ca.pi)
    ocp.set_lims(d, -0.4, 0.4)

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
    th00.append(0)
    ocp.set_inital(th, th00.copy())
    v00 = np.array([lib.get_distance(wpts[:, i], wpts[:, i+1]) for i in range(N1)]) / dt00
    ocp.set_inital(v, v00)
    th00.pop(-1)
    d00 = np.arctan(np.array(th00) * 0.33 / v00)
    ocp.set_inital(d, d00)

    ocp.set_up_solve()

    plt.figure(1)
    plt.clf()
    plt.plot(wpts[0, :], wpts[1, :], 'o', markersize=12)

    plt.plot(x00, y00, '+', markersize=20)

    plt.pause(0.5)
    # plt.show()

    for i in range(20):
        wpts = np.array(wp[:, i:i+N])
        x0 = [wpts[0, 0], wpts[1, 0], lib.get_bearing(wpts[:, 0], wpts[:, 1])]

        ocp.set_objective(x, wpts[0, :])
        ocp.set_objective(y, wpts[1, :])

        states, controls, times = ocp.solve(x0)

        xs = states[:N]
        ys = states[N:2*N]
        ths = states[2*N:]
        vs = controls[:N1]
        ds = controls[N1:]
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

        # plt.pause(0.5)
        plt.show()

if __name__ == "__main__":
    example_loop_v_d()