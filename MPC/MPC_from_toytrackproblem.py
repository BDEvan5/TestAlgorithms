import numpy as np
import casadi as ca 
from matplotlib import pyplot as plt 

import LibFunctions as lib 


# def ModelPredictiveController(ts, nvecs, ns):
#     offsets = nvecs * ns # check multiplication works
#     pts = lib.add_locations(ts, offsets)
#     th_ns = [lib.get_bearing([0, 0], nvecs[i, 0:2]) for i in range(len(nvecs))]

def ModelPredictiveController(pts):
    pts = np.array(pts)

    x00 = pts[0, 0]
    y00 = pts[0, 1]
    v00 = 2

    N = len(pts)
    N1 = N-1

    x_f = ca.MX.sym('x_f', N)
    y_f = ca.MX.sym('y_f', N)

    x0_f = ca.MX.sym('x0_f', N-1)
    x1_f = ca.MX.sym('x1_f', N-1)
    y0_f = ca.MX.sym('y0_f', N-1)
    y1_f = ca.MX.sym('y1_f', N-1)

    dis = ca.Function('dis', [x0_f, x1_f, y0_f, y1_f], [ca.sqrt((x1_f-x0_f)**2 + (y1_f-y0_f)**2)])
    
    track_length = ca.Function('length', [x_f, y_f], [dis(x_f[1:], x_f[:-1], y_f[1:], y_f[:-1])])
   
    # state helpers
    d_f = ca.MX.sym('d_f', N-2)
    v_f1 = ca.MX.sym('d_f', N-1)
    v_f2 = ca.MX.sym('d_f', N-2)
    d_th = ca.Function('d_th', [d_f, v_f2], [v_f2 / 0.33 * ca.tan(d_f)])
    d_dt = ca.Function('d_dt', [v_f1, x_f, y_f], [track_length(x_f, y_f) / v_f1])


    # define symbols
    x = ca.MX.sym('x', N)
    y = ca.MX.sym('y', N)
    th = ca.MX.sym('th', N-1)
    dt = ca.MX.sym('dt', N-1)
    v = ca.MX.sym('v', N-1)
    d = ca.MX.sym('d', N-1)

    nlp = {\
    'x': ca.vertcat(x, y, th, dt, v, d),
    'f': ca.sumsqr(x - pts[:, 0]) + ca.sumsqr(y - pts[:, 0]) + ca.sumsqr(dt), 
    'g': ca.vertcat(
                # dynamic constraints
                x[1:] - (x[:-1] + v * ca.cos(th)),
                y[1:] - (y[:-1] + v * ca.sin(th)),
                th[1:] - (th[:-1] + dt[:-1] * d_th(d[:-1], v[:-1])),
                dt - d_dt(v, x, y),

                # boundary constraints
                x[0] - x00, y[0] - y00,
                # x[0] - pts[-1, 0], y[0] - pts[-1, 1],
                v[0] - v00
            ) \
    
    }

    S = ca.nlpsol('S', 'ipopt', nlp, {'ipopt':{'print_level':5}})
    # S = ca.nlpsol('S', 'ipopt', nlp, {'ipopt':{'print_level':0}})

    v0 = [v00] * N1

    th0 = []
    for i in range(N-1):
        th_00 = lib.get_bearing(pts[i], pts[i+1])
        th0.append(th_00)

    th0 = np.array(th0)

    dth0 = []
    for i in range(N-2):
        dth = lib.sub_angles_complex(th0[i+1], th0[i])
        dth0.append(dth)

    dth0 = np.array(dth0)
    d0 = np.arctan(dth0 * 0.33 / v0[:-1])
    d0 = np.append(d0, 0)

    dt0 = d_dt(v0, pts[:, 0], pts[:, 1])

    x0 = ca.vertcat(pts[:, 0], pts[:, 1], th0, dt0, v0, d0)

    d_max = 0.4
    v_max = 7.5
    lbx = [0] * N + [0] * N + [-np.pi]*N1 + [0] * N1 + [0] * N1+ [-d_max] * N1
    ubx = [10] * N + [10] * N + [np.pi]*N1 + [2] * N1 + [v_max] * N1 + [d_max] * N1

    plt.figure(1)
    plt.plot(pts[:, 0], pts[:, 1], '+', markersize=16)
    plt.pause(0.0001)
    

    r = S(x0=x0, lbg=0, ubg=0, lbx=lbx, ubx=ubx)

    x_opt = r['x']

    x = np.array(x_opt[:N])
    y = np.array(x_opt[N:2*N])
    thetas = np.array(x_opt[2*N:2*N + N1])
    dt = np.array(x_opt[2*N + N1:2*N + N1*2])
    v = np.array(x_opt[2*N + 2*N1:2*N + N1*3])
    d = np.array(x_opt[2*N + 3*N1:2*N + N1*4])


    plt.figure(1)
    plt.plot(pts[:, 0], pts[:, 1], '+', markersize=16)
    plt.plot(x, y, 'x', markersize=16)

    plt.pause(0.001)

    plt.figure(2)
    plt.plot(v)
    

    plt.show()

# testing
def main_test():
    pts = [[1, 1], [1.2, 1.1], [1.3, 1.2], [1.5, 1.5], [1.7, 1.6]]

    ModelPredictiveController(pts)    




if __name__ == "__main__":
    main_test()

