import numpy as np 
import casadi as ca


class Ocp:
    def __init__(self):
        self.states = []
        self.controls = []

        self.state_der = {}
        self.initial = {}

    def create_state(self):
        state = ca.MX.sym("x")
        self.states.append(state)

        return state

    def control(self):
        control = ca.MX.sym("x")
        self.states.append(control)

        return control

    def set_der(self, state, der):
        self.state_der[state] = der

    def set_inital(self, state, init_val):
        self.initial[state] = init_val

    def solve(self):

        nlp = {\
            'x': ca.vertcat(state for state in self.states),
            'f': 
            'g': 

            }




def example():
    ocp = Ocp()

    x = ocp.create_state()

    x_dot = ocp.control()

    ocp.set_der(x, x_dot)