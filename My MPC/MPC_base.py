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

