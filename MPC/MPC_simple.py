import casadi as ca 
import numpy as np 
import matplotlib.pyplot as plt 

class CasadiMPC:
    def __init__(self):
        T = 10 # time horizon
        N = 20 # control pts

        x1 = ca.SX.sym('x1')
        x2 = ca.SX.sym('x2')
        u = ca.SX.sym('u')

        xdot = [(1-x2^2)*x1 - x2 + u, x1]

        # objective
        L = x1^2 + x2^2 + u^2

        # dynamics
        f = ca.Function('f', [x, u], [xdot, L])

        M = 4
        DT = T/N/M
        # repeats f
        x0 = ca.MX.sym('x0', 2)
        u0 = ca.MX.sym('u0', 2)
        x = x0
        Q = 0
        for i in range(M):
            k1, k1_q = f(X, U)
            k2, k2_q = f(X + DT/2 * k1, U)
            k3, k3_q = f(X + DT/2 * k2, U)
            k4, k4_q = f(X + DT * k3, U)
            X=X+DT/6*(k1 +2*k2 +2*k3 +k4)
            Q = Q + DT/6*(k1_q + 2*k2_q + 2*k3_q + k4_q)

        F = ca.Function('F', [x0, u0], [x, Q], ['x0','p'], ['xf', 'qf'])

        w = {}
        w0 = []
        lbw = []
        ubg = []
        J = 0
        g = {}
        lbg = []
        ubg = []

        x0 = ca.MX.sym('x0', 2)
        w = {w{:}, x0}
        lbw = [lbw, 0, 1]
        ubw = [ubw, 0, 1]
        w0 = [w0, 0, 1]

        Xk = x0
        for k in range(N-1):
            Uk = ca.MX.sym(f"u_{k}")
            w = {w{:}, Uk}
            lbw = [lbw, -1]
            ubw = [ubw, 1]
            w0 = [w0, 0]

            Fk = F('x0', Xk, 'p', 'Uk')
            Xk_end = Fk.xf
            J = J + Fk.qf

            Xk = ca.MX.sym(f"X_{k+1}", 2)
            w = {w{:}, Xk}
            lbw = [lbw, -0.25, -ca.inf]
            ubw = [ubw, ca.inf, -ca.inf]
            w0 = [w0, 0, 0]

            g = {g{:}, Xk_end-Xk}
            lbg = [lbg, 0, 0]
            ubg = [ubg, 0, 0]

        prob = {
            'x': ca.vertcat(w{:}),
            'f': J,
            'g': ca.vertcat(g{:})
        }

        S = ca.nlpsol('S', 'ipopt', nlp)

        self.solver = S
        self.x0 = x0
        self.lbx = lbw
        self.ubx = ubw
        self.lbg = lbg
        self.ubg = ubg

    def stepImpl(self, x, t):
        w0 = self.w0
        lbw = self.lbx
        ubw = self.ubx
        solver = self.solver
        lbw[0:1] = x
        ubw[0:1] = x

        sol = solver(x0=w0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

        out = sol['x']





