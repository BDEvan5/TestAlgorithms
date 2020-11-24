import numpy as np 
import casadi as ca
from matplotlib import pyplot as plt


class Ocp:
    def __init__(self, n_pts):
        self.n = n_pts
        self.states = []
        self.controls = []

        self.state_der = {}
        self.initial = {}
        self.constraints = {}
        self.objectives = {}

        self.dt = ca.MX.sym('dt', self.n-1)
        # self.states.append(dt)

    def create_state(self, name="x"):
        state = ca.MX.sym(name, self.n)
        self.states.append(state)

        return state

    def control(self, name="u"):
        control = ca.MX.sym(name, self.n-1)
        self.controls.append(control)

        return control

    def set_der(self, state, der):
        self.state_der[state] = der

    def set_inital(self, state, init_val):
        self.initial[state] = init_val

    def set_objective(self, var, objective):
        self.objectives[var] = objective

    def set_constraints(self, state, constraint):
        self.constraints[state] = constraint

    def solve(self):
        N = self.n
        N1 = N - 1

        xs = [state for state in self.states]
        us = [control for control in self.controls]

        dyns = [var[1:] - (var[:-1] + self.state_der[var] * self.dt) for var in self.states]
        cons = [cons[0] - self.constraints[cons] for cons in self.constraints.keys()]

        obs = [o - self.objectives[o] for o in self.objectives.keys()]

        nlp = {\
            'x': ca.vertcat(xs[0], us[0], self.dt),
            'f': ca.sumsqr(obs[0]),
            'g': ca.vertcat(dyns[0], cons[0])
            }

        max_speed = 1

        lbx = [0]  * N + [-max_speed]  * N1 + [0] * N1
        ubx = [ca.inf]  * N + [max_speed]  * N1 + [10] * N1

        lbg = [0] * N 
        ubg = [0] * N 

        x00 = [self.initial[state] for state in self.states]
        u00 = [self.initial[control] for control in self.controls]
        x0 = ca.vertcat(x00[0], u00[0], self.initial[self.dt])
        
        S = ca.nlpsol('S', 'ipopt', nlp)
        r = S(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        x_opt = r['x']

        return x_opt

    




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

    x = ocp.create_state('x')
    x_dot = ocp.control('x_dot')

    ocp.set_der(x, x_dot)
    ocp.set_objective(x, wpts[0, :])
    ocp.set_objective(ocp.dt, ca.GenMX_zeros(N1))
    ocp.set_constraints(x, wpts[0, 0])

    x00 = wpts[0, :]
    T = 5
    dt00 = [T/N1] * N1
    xd00 = (x00[1:] - x00[:-1]) / dt00
    ocp.set_inital(x, x00)
    ocp.set_inital(x_dot, xd00)
    ocp.set_inital(ocp.dt, dt00)

    x_opt = ocp.solve()

    x = np.array(x_opt[:N])
    xds = np.array(x_opt[N:N + N1])
    times = np.array(x_opt[N + N1:])
    total_time = np.sum(times)

    print(f"Times: {times}")
    print(f"Total Time: {total_time}")
    print(f"X dots: {xds.T}")
    print(f"xs: {x.T}")

    print(f"----------------")
    print(f"{xds * times}")

    plt.figure(1)
    plt.plot(wpts[0, :], np.ones_like(wpts[0, :]), 'o', markersize=12)

    plt.plot(x, np.ones_like(x), '+', markersize=20)

    plt.show()



if __name__ == "__main__":
    example()

